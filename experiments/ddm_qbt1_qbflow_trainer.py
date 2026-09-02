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
import tarfile
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
from tac.witness_control.birth_completion import DEFAULT_TAU_PERSIST
from tac.witness_dsl.curriculum_dsl import EmaDecayCalibrated

SCHEMA = "ddm_qbt1_qbflow_compiled_config.v1"
CHECKPOINT_SCHEMA = "ddm_qbt1_qbflow_checkpoint.v1"
RESULT_SCHEMA = "ddm_qbflow_observability.v1"
CONTROL_SCHEMA = "ddm_qbt1_same_budget_qbw1_control.v1"
MEMORY_SCHEMA = "ddm_qbt1_materialization_memory_projection.v1"
LAUNCH_SCHEMA = "ddm_qbt1_compiled_launch_request.v1"
PALETTE_INIT_SCHEMA = "ddm_qbt2b_inherited_palette_init.v1"
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
QBT2B_ROOT = TRAIN_ROOT / "qbt2b_inherited_palette_birth"
R7_RETENTION_ROOT = Path("/Volumes/APDataStore/pact/ddm_qbt2b_r7_lane_constrained_margin")
QBR1_RETENTION_ROOT = Path("/Volumes/APDataStore/pact/ddm_qbr1_born_fairform_burn_prep")
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
FP1_PALETTE = Path("/Volumes/VertigoDataTier/pact/ddm_fp1_20260731/prototypes.npz")
R1_CHECKPOINT = (
    TRAIN_ROOT
    / "governed_n32/stage_05_same_budget_admission/checkpoints/stage_05_end.pt"
)
R1_RETAINED = TRAIN_ROOT / "governed_n32/stage_05_same_budget_admission/retained"
R2_REPACK_REFERENCE_TAR = (
    TRAIN_ROOT
    / "governed_n32_r2/stage_03_joint_boundary_interior_birth/reencoded/step_004860.tar"
)
R2_FIRST_CHECKPOINT = (
    TRAIN_ROOT
    / "governed_n32_r2/stage_03_joint_boundary_interior_birth/checkpoints/periodic_step_000005.pt"
)
R2_FINAL_CHECKPOINT = (
    TRAIN_ROOT
    / "governed_n32_r2/stage_03_joint_boundary_interior_birth/checkpoints/periodic_step_004865.pt"
)
FP1_PALETTE_SHA256 = "19e6524b75724f0b19f0e2e49a827d9f28b40d087b1e5504c3a85577a9e76f0b"
R1_CHECKPOINT_SHA256 = "7e3fb97199eca5cd03eac8c2b858bdcea3b6ddad83eba9c410161252455084cd"
PALETTE_CLASSES = ("Road", "Lane", "Undrivable", "Movable", "MyCar")
READOUT_FIT_SAMPLES_PER_PAIR_CLASS = 512
READOUT_RESIDUAL_VARIANCE_RATIO_MAX = 0.95
BIRTH_WITHIN_CLASS_ERROR_MAX = 1.0 - DEFAULT_TAU_PERSIST
# r6 gate revision (ddm_qbt2b_r5_balanced_ce_verdict_20260828.md §5): the birth EVENT is
# EXISTENCE — every class majority-correct (within-class error < 0.50) means each class's
# plurality basin exists in the realized argmax, so the expected-flip margin objective has
# gradient support at every class boundary (the precondition r2's frozen run lacked). The
# legacy accuracy bar (werr < 0.20) demotes to a WATCH metric under this mode; the
# validator pins (mode, threshold) as a PAIR so a loosened threshold cannot ride in under
# the legacy mode label.
BIRTH_EXISTENCE_ERROR_MAX = 0.50
BIRTH_EVENT_MODE_THRESHOLDS = {
    "accuracy_020": BIRTH_WITHIN_CLASS_ERROR_MAX,
    "existence_majority": BIRTH_EXISTENCE_ERROR_MAX,
}
# r7 constrained-margin law (ddm_qbt2b_r6_born_field_margin_verdict_20260828.md
# §§4-5).  Bounds preserve the born r5/r6 handoff field rather than shifting its
# optimum: Lane was 0.0980 at the r5 endpoint and 0.116328/0.119581 at the two
# retained r6 birth verdicts, so 0.12 is their measured upper envelope rounded
# outward; Movable was 0.0065 at r5 and 0.007490/0.008856 at r6 birth, so its
# outward envelope is 0.009.  The shared eta is derived from the retained r6
# endpoint Lane werr 0.9981336319522209: one natural class-loss unit after ten
# persistent endpoint-sized violations, eta = 1/(10*(0.9981336319522209-0.12)).
# lambda_max=5 reuses the reviewed ddm_lg1 (#808) natural-unit ceiling; reaching
# it is a fail-loud infeasibility signal, never a license to dominate the primal.
MARGIN_CONSTRAINT_UNCONSTRAINED = "unconstrained"
MARGIN_CONSTRAINT_LANE_MOVABLE = "lane_movable_werr_primal_dual"
MARGIN_CONSTRAINT_LANE_BOUND = 0.12
MARGIN_CONSTRAINT_MOVABLE_BOUND = 0.009
MARGIN_CONSTRAINT_ETA_LAMBDA = 0.11387788414126129
MARGIN_CONSTRAINT_LAMBDA_MAX = 5.0
MARGIN_CONSTRAINT_MODE_PINS = {
    MARGIN_CONSTRAINT_UNCONSTRAINED: {
        "bounds": {},
        "eta_lambda": 0.0,
    },
    MARGIN_CONSTRAINT_LANE_MOVABLE: {
        "bounds": {
            "Lane": MARGIN_CONSTRAINT_LANE_BOUND,
            "Movable": MARGIN_CONSTRAINT_MOVABLE_BOUND,
        },
        "eta_lambda": MARGIN_CONSTRAINT_ETA_LAMBDA,
    },
}
QBT2B_BIRTH_MAX_STEPS = 100
QBT2B_MARGIN_STEPS = 5_000
QBT2B_TOTAL_STEPS = QBT2B_BIRTH_MAX_STEPS + QBT2B_MARGIN_STEPS

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
    # Deflate the retained payloads (measured 2.07x on real verdict frames);
    # np.load reads STORED and DEFLATED members identically, arrays unchanged.
    np.savez_compressed(payload, **arrays)
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
    allowed_roots = (STORE.resolve(), R7_RETENTION_ROOT.resolve(), QBR1_RETENTION_ROOT.resolve())
    if not any(resolved == root or root in resolved.parents for root in allowed_roots):
        raise QBT1Error(
            "QBT1 output must remain under an authorized AP custody root: "
            + ", ".join(map(str, allowed_roots))
        )
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
            "render_state": render_state.reshape(batch, height, width, -1),
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


def per_class_expected_flip_margin_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    tau: float,
    class_id: int,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Expected flip restricted to real target pixels of one scorer class."""

    if logits.ndim != 4 or target.shape != (logits.shape[0], logits.shape[2], logits.shape[3]):
        raise QBT1Error("per-class expected-flip logits/target geometry differs")
    if not tau > 0 or not 0 <= int(class_id) < logits.shape[1]:
        raise QBT1Error("per-class expected-flip tau/class differs")
    target_index = target[:, None].long()
    target_logit = logits.gather(1, target_index).squeeze(1)
    other = logits.clone()
    other.scatter_(1, target_index, -1.0e9)
    flip_probability = torch.sigmoid(-(target_logit - other.amax(dim=1)) / tau)
    target_mask = target == int(class_id)
    if not bool(target_mask.any()):
        raise QBT1Error(f"per-class expected-flip target class is absent: {class_id}")
    pixel_weights = target_mask.to(flip_probability)
    if sample_weights is not None:
        weights = sample_weights.to(flip_probability)
        if weights.shape != (logits.shape[0],) or not bool(torch.all(weights > 0)):
            raise QBT1Error("per-class expected-flip sample weights differ")
        pixel_weights = pixel_weights * weights[:, None, None]
    return (flip_probability * pixel_weights).sum() / pixel_weights.sum()


def realized_within_class_error(
    logits: torch.Tensor, target: torch.Tensor, class_id: int
) -> float:
    """Realized render->R->uint8->SegNet argmax error on target class pixels."""

    if logits.ndim != 4 or target.shape != (logits.shape[0], logits.shape[2], logits.shape[3]):
        raise QBT1Error("realized within-class logits/target geometry differs")
    target_mask = target == int(class_id)
    if not bool(target_mask.any()):
        raise QBT1Error(f"realized within-class target class is absent: {class_id}")
    predicted = logits.detach().argmax(dim=1)
    # CPU hop BEFORE the float64 cast: MPS has no float64, and the realized werr
    # must be the exact float64 mean regardless of the training device.
    return float((predicted[target_mask] != int(class_id)).cpu().to(torch.float64).mean())


def dual_ascent_margin_constraints(
    lambdas: Mapping[str, float],
    realized_werr: Mapping[str, float],
    bounds: Mapping[str, float],
    *,
    eta_lambda: float,
    lambda_max: float = MARGIN_CONSTRAINT_LAMBDA_MAX,
) -> dict[str, float]:
    """Projected per-class dual ascent on realized within-class error constraints."""

    if set(lambdas) != set(bounds) or set(realized_werr) != set(bounds):
        raise QBT1Error("margin-constraint class sets differ")
    if eta_lambda <= 0.0 or lambda_max <= 0.0:
        raise QBT1Error("margin-constraint dual geometry differs")
    updated: dict[str, float] = {}
    for class_name, bound in bounds.items():
        previous = float(lambdas[class_name])
        werr = float(realized_werr[class_name])
        if not 0.0 <= previous <= lambda_max or not 0.0 <= float(bound) < 1.0:
            raise QBT1Error("margin-constraint state/bound differs")
        if not 0.0 <= werr <= 1.0:
            raise QBT1Error("realized within-class error is outside [0,1]")
        updated[class_name] = max(
            0.0,
            min(lambda_max, previous + eta_lambda * (werr - float(bound))),
        )
    return updated


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
    margin_constraint_lambdas: Mapping[str, float] | None = None,
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
    components = {
        "loss_total": total,
        "seg_expected_flip_realized": realized_seg,
        "seg_expected_flip_native_interface": interface_seg,
        "pose_mse_realized": pose_mse,
        "pose_score_realized": pose_score,
        "tau": total.new_tensor(tau),
        "camera_min": camera_pair.detach().amin(),
        "camera_max": camera_pair.detach().amax(),
    }
    if margin_constraint_lambdas is not None:
        expected_names = set(MARGIN_CONSTRAINT_MODE_PINS[MARGIN_CONSTRAINT_LANE_MOVABLE]["bounds"])
        if set(margin_constraint_lambdas) != expected_names:
            raise QBT1Error("margin-constraint lambda class set differs")
        penalty = total.new_zeros(())
        for class_name, class_id in (("Lane", 1), ("Movable", 3)):
            class_flip = per_class_expected_flip_margin_loss(
                scorer_logits,
                target_argmax,
                tau,
                class_id,
                sample_weights,
            )
            class_penalty = 100.0 * float(margin_constraint_lambdas[class_name]) * class_flip
            penalty = penalty + class_penalty
            components[f"margin_constraint_expected_flip_{class_name}"] = class_flip
            components[f"margin_constraint_penalty_score_{class_name}"] = class_penalty
        total = total + penalty
        components["margin_constraint_penalty_score"] = penalty
        components["loss_total"] = total
    return total, components


def derive_balanced_class_weights(pair_ids: Sequence[int], device: torch.device) -> torch.Tensor:
    """Balanced inverse-frequency class weights derived from the sealed selection's REAL targets.

    w_c = total_pixels / (num_classes * count_c), so a class carrying f of the pixel mass
    receives 1/(K*f) gradient scale — the standard balanced heuristic, derived at runtime
    from the actual GT (no hand-typed area constants). Fails closed if any class is absent
    from the selection, because a zero-count weight would be infinite.
    """

    target_argmax, _unused_pose = _target_arrays(pair_ids, torch.device("cpu"))
    counts = torch.bincount(target_argmax.reshape(-1), minlength=qbf1.N_CLASSES).to(torch.float64)
    if int((counts == 0).sum()) != 0:
        raise QBT1Error("balanced class weights need every class present in the selection targets")
    weights = counts.sum() / (float(qbf1.N_CLASSES) * counts)
    return weights.to(dtype=torch.float32, device=device)


def realized_ce_birth_objective(
    camera_pair: torch.Tensor,
    scorer_pose6: torch.Tensor,
    scorer_logits: torch.Tensor,
    target_argmax: torch.Tensor,
    target_pose6: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
    class_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Class-birth CE through the real scorer path, with pose active from step zero."""

    per_pixel = F.cross_entropy(
        scorer_logits, target_argmax.long(), weight=class_weights, reduction="none"
    )
    if class_weights is None:
        ce_per_sample = per_pixel.mean(dim=(1, 2))
    else:
        pixel_weight = class_weights[target_argmax.long()]
        ce_per_sample = per_pixel.sum(dim=(1, 2)) / pixel_weight.sum(dim=(1, 2))
    pose_per_sample = (scorer_pose6 - target_pose6).square().mean(dim=1)
    if sample_weights is None:
        realized_ce = ce_per_sample.mean()
        pose_mse = pose_per_sample.mean()
    else:
        weights = sample_weights.to(ce_per_sample)
        if weights.shape != ce_per_sample.shape or not bool(torch.all(weights > 0)):
            raise QBT1Error("birth-stage sample weights differ")
        realized_ce = (ce_per_sample * weights).sum() / weights.sum()
        pose_mse = (pose_per_sample * weights).sum() / weights.sum()
    pose_score = torch.sqrt(torch.clamp(10.0 * pose_mse, min=1.0e-20))
    total = 100.0 * realized_ce + pose_score
    return total, {
        "loss_total": total,
        "seg_ce_realized": realized_ce,
        "pose_mse_realized": pose_mse,
        "pose_score_realized": pose_score,
        "camera_min": camera_pair.detach().amin(),
        "camera_max": camera_pair.detach().amax(),
    }


def birth_gate_from_table(
    per_class: Sequence[Mapping[str, Any]],
    *,
    within_class_error_max: float = BIRTH_WITHIN_CLASS_ERROR_MAX,
) -> dict[str, Any]:
    if len(per_class) != qbf1.N_CLASSES or not 0.0 < within_class_error_max < 1.0:
        raise QBT1Error("birth gate geometry differs")
    classes = []
    for row in per_class:
        within_error = row.get("within_class_error")
        passed = bool(
            int(row.get("predicted_pixels", 0)) > 0
            and within_error is not None
            and float(within_error) < within_class_error_max
        )
        classes.append(
            {
                "class_id": int(row["class_id"]),
                "class_name": str(row["class_name"]),
                "predicted_pixel_share": float(row["predicted_pixel_share"]),
                "within_class_error": within_error,
                "passed": passed,
            }
        )
    if math.isclose(
        within_class_error_max, BIRTH_EXISTENCE_ERROR_MAX, rel_tol=0.0, abs_tol=1.0e-12
    ):
        derived_from = (
            "majority-correct existence event: within-class error < 0.50 means the class's "
            "plurality basin exists in the realized argmax, so the expected-flip margin "
            "objective has gradient support at every class boundary "
            "(ddm_qbt2b_r5_balanced_ce_verdict_20260828.md §5)"
        )
    else:
        derived_from = (
            "1 - tac.witness_control.birth_completion.DEFAULT_TAU_PERSIST "
            f"({DEFAULT_TAU_PERSIST})"
        )
    gate = {
        "derived_from": derived_from,
        "within_class_error_max_exclusive": within_class_error_max,
        "all_five_classes_pass": all(row["passed"] for row in classes),
        "classes": classes,
    }
    if within_class_error_max > BIRTH_WITHIN_CLASS_ERROR_MAX:
        # The legacy accuracy bar rides along as a WATCH metric (never a gate) so
        # margin-stage sharpening stays observable in every existence-mode verdict.
        gate["accuracy_watch"] = {
            "error_max_exclusive": BIRTH_WITHIN_CLASS_ERROR_MAX,
            "classes_passing": sum(
                1
                for row in classes
                if row["within_class_error"] is not None
                and float(row["within_class_error"]) < BIRTH_WITHIN_CLASS_ERROR_MAX
            ),
        }
    return gate


def evaluate_birth_verdict(
    root: Path,
    *,
    model: QBFLOWTorch,
    ema: EMA,
    posenet: nn.Module,
    segnet: nn.Module,
    pair_ids: Sequence[int],
    chunk_pairs: int,
    step: int,
    verdict_index: int,
    within_class_error_max: float,
) -> dict[str, Any]:
    """Evaluate and retain one event verdict on the realized argmax surface."""

    arrays: dict[str, list[np.ndarray]] = {
        "camera_pair_u8": [],
        "segnet_logits_f16": [],
        "segnet_argmax_u8": [],
        "target_argmax_u8": [],
        "posenet_pose6_f32": [],
        "target_pose6_f32": [],
    }
    with ema_scope(model, ema), torch.no_grad():
        for chunk_ids in pair_chunks(pair_ids, chunk_pairs):
            ids_tensor = torch.tensor(chunk_ids, dtype=torch.long, device=next(model.parameters()).device)
            target_argmax, target_pose6 = _target_arrays(chunk_ids, ids_tensor.device)
            outputs = model(ids_tensor, height=EVAL_H, width=EVAL_W)
            camera = roundtrip_to_camera_uint8_ste(outputs["rgb_pair_01"])
            pose6, logits = scorer_forward(camera, posenet, segnet)
            arrays["camera_pair_u8"].append(
                camera.round().clamp(0, 255).cpu().to(torch.uint8).numpy()
            )
            arrays["segnet_logits_f16"].append(logits.cpu().numpy().astype("<f2"))
            arrays["segnet_argmax_u8"].append(
                logits.argmax(dim=1).cpu().numpy().astype(np.uint8)
            )
            arrays["target_argmax_u8"].append(target_argmax.cpu().numpy().astype(np.uint8))
            arrays["posenet_pose6_f32"].append(pose6.cpu().numpy().astype("<f4"))
            arrays["target_pose6_f32"].append(target_pose6.cpu().numpy().astype("<f4"))
    retained_arrays = {name: np.concatenate(parts) for name, parts in arrays.items()}
    retained_arrays["pair_ids_i64"] = np.asarray(pair_ids, dtype=np.int64)
    payload = atomic_npz(
        root / f"verdict_{verdict_index:04d}_step_{step:06d}.npz", **retained_arrays
    )
    table = per_class_argmax_table(
        retained_arrays["segnet_argmax_u8"], retained_arrays["target_argmax_u8"]
    )
    pose_mse = float(
        np.square(
            retained_arrays["posenet_pose6_f32"].astype(np.float64)
            - retained_arrays["target_pose6_f32"].astype(np.float64)
        ).mean()
    )
    return {
        "schema": "ddm_qbt2b_realized_birth_verdict.v1",
        "axis": "[macOS frozen-scorer advisory]",
        "score_claim": False,
        "step": int(step),
        "verdict_index": int(verdict_index),
        "pair_ids": list(map(int, pair_ids)),
        "per_class": table,
        "pose_mse_realized": pose_mse,
        "gate": birth_gate_from_table(
            table, within_class_error_max=within_class_error_max
        ),
        "retained_payload": payload,
        "all_materialized_frames_and_scorer_outputs_retained": True,
    }


def load_initial_model(device: torch.device) -> QBFLOWTorch:
    with np.load(INITIAL_PARAMS, allow_pickle=False) as params_npz:
        params = {name: np.asarray(params_npz[name], dtype=np.float32) for name in params_npz.files}
    with np.load(INITIAL_LATENTS, allow_pickle=False) as latents_npz:
        boundary = np.asarray(latents_npz["boundary"], dtype=np.float32)
        interior = np.asarray(latents_npz["interior"], dtype=np.float32)
    return QBFLOWTorch(params, boundary, interior).to(device)


def verify_qbt2b_inputs() -> dict[str, dict[str, Any]]:
    """Verify the two inherited, video-derived inputs without changing QBF1 pins."""

    rows = {
        "fp1_palette": file_fact(FP1_PALETTE),
        "qbt1_r1_checkpoint": file_fact(R1_CHECKPOINT),
    }
    expected = {
        "fp1_palette": FP1_PALETTE_SHA256,
        "qbt1_r1_checkpoint": R1_CHECKPOINT_SHA256,
    }
    for name, row in rows.items():
        if row["sha256"] != expected[name]:
            raise QBT1Error(f"QBT2B inherited input drifted: {name}")
    return rows


def load_fp1_inherited_palette() -> tuple[np.ndarray, np.ndarray]:
    with np.load(FP1_PALETTE, allow_pickle=False) as payload:
        if "proto_solved" not in payload or "sample_ids" not in payload:
            raise QBT1Error("FP1 palette artifact lacks the documented trained values or sample IDs")
        palette = np.asarray(payload["proto_solved"], dtype=np.float32).copy()
        sample_ids = np.asarray(payload["sample_ids"], dtype=np.int64).copy()
    if palette.shape != (qbf1.N_CLASSES, 3) or not np.isfinite(palette).all():
        raise QBT1Error("FP1 inherited palette geometry differs")
    if float(palette.min()) < 0.0 or float(palette.max()) > 255.0:
        raise QBT1Error("FP1 inherited palette lies outside uint8 RGB range")
    if sample_ids.shape != (32,) or len(set(map(int, sample_ids))) != 32:
        raise QBT1Error("FP1 palette training sample lineage differs")
    return palette, sample_ids


def load_r1_ema_model(device: torch.device) -> tuple[QBFLOWTorch, Mapping[str, Any]]:
    verify_qbt2b_inputs()
    payload = torch.load(R1_CHECKPOINT, map_location="cpu", weights_only=False)
    if payload.get("schema") != CHECKPOINT_SCHEMA or "ema" not in payload:
        raise QBT1Error("QBT1 r1 checkpoint schema or EMA payload differs")
    model = load_initial_model(device)
    shadow = payload["ema"].get("shadow")
    if set(shadow or ()) != set(model.state_dict()):
        raise QBT1Error("QBT1 r1 EMA tensor set differs from the immutable QBF1 twin")
    model.load_state_dict(
        {name: value.detach().clone().to(device) for name, value in shadow.items()}, strict=True
    )
    return model, payload


def _evenly_spaced_rows(indices: np.ndarray, maximum: int) -> np.ndarray:
    if indices.size <= maximum:
        return indices
    positions = np.linspace(0, indices.size - 1, num=maximum, dtype=np.int64)
    return indices[positions]


def collect_readout_fit_samples(
    model: QBFLOWTorch,
    pair_ids: Sequence[int],
    *,
    samples_per_pair_class: int = READOUT_FIT_SAMPLES_PER_PAIR_CLASS,
) -> dict[str, np.ndarray]:
    """Collect deterministic observed render states stratified by pair and native class."""

    if samples_per_pair_class < 1:
        raise QBT1Error("readout fit sample cap must be positive")
    states: list[np.ndarray] = []
    classes: list[np.ndarray] = []
    pairs: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for chunk_ids in pair_chunks(pair_ids, 4):
            outputs = model(torch.tensor(chunk_ids, dtype=torch.long), height=EVAL_H, width=EVAL_W)
            render_state = outputs["render_state"].detach().cpu().numpy().astype(np.float32)
            native_class = outputs["class_logits"].argmax(dim=-1).detach().cpu().numpy()
            for local_index, pair_id in enumerate(chunk_ids):
                flat_state = render_state[local_index].reshape(-1, render_state.shape[-1])
                flat_class = native_class[local_index].reshape(-1)
                for class_id in range(qbf1.N_CLASSES):
                    indices = np.flatnonzero(flat_class == class_id)
                    chosen = _evenly_spaced_rows(indices, samples_per_pair_class)
                    if chosen.size == 0:
                        continue
                    states.append(flat_state[chosen].copy())
                    classes.append(np.full(chosen.size, class_id, dtype=np.uint8))
                    pairs.append(np.full(chosen.size, pair_id, dtype=np.uint16))
    if not states:
        raise QBT1Error("QBT1 r1 produced no native regions for the readout fit")
    return {
        "render_state_f32": np.concatenate(states, axis=0),
        "native_class_u8": np.concatenate(classes, axis=0),
        "pair_id_u16": np.concatenate(pairs, axis=0),
    }


def fit_inherited_palette_readout(
    model: QBFLOWTorch,
    palette_rgb: np.ndarray,
    pair_ids: Sequence[int],
    *,
    samples: Mapping[str, np.ndarray] | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Fit only last-frame render readout values to the CE-trained inherited palette."""

    samples = dict(samples) if samples is not None else collect_readout_fit_samples(model, pair_ids)
    states = samples["render_state_f32"].astype(np.float64)
    classes = samples["native_class_u8"].astype(np.int64)
    clipped = np.clip(np.asarray(palette_rgb, dtype=np.float64) / 255.0, 1.0e-5, 1.0 - 1.0e-5)
    palette_logits = np.log(clipped / (1.0 - clipped))
    targets = palette_logits[classes]
    design = np.concatenate((states, np.ones((states.shape[0], 1), dtype=np.float64)), axis=1)
    coefficients, _residuals, rank, singular_values = np.linalg.lstsq(
        design, targets, rcond=1.0e-6
    )
    predicted_logits = np.einsum("ni,ij->nj", design, coefficients, optimize=False)
    predicted_rgb = 255.0 / (1.0 + np.exp(-predicted_logits))
    residual = predicted_logits - targets
    total_variance = float(np.square(targets - targets.mean(axis=0, keepdims=True)).sum())
    residual_variance = float(np.square(residual).sum())
    variance_ratio = residual_variance / total_variance if total_variance > 0.0 else math.inf
    class_rows = []
    for class_id, class_name in enumerate(PALETTE_CLASSES):
        mask = classes == class_id
        class_rows.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "sample_count": int(mask.sum()),
                "target_rgb": palette_rgb[class_id].astype(float).tolist(),
                "predicted_rgb_mean": predicted_rgb[mask].mean(axis=0).astype(float).tolist()
                if bool(mask.any())
                else None,
                "rgb_rmse": np.sqrt(
                    np.square(predicted_rgb[mask] - palette_rgb[class_id]).mean(axis=0)
                ).astype(float).tolist()
                if bool(mask.any())
                else None,
                "logit_rmse": np.sqrt(np.square(residual[mask]).mean(axis=0)).astype(float).tolist()
                if bool(mask.any())
                else None,
            }
        )
    expected_rank = design.shape[1]
    degenerate_reasons = []
    if int(rank) < expected_rank:
        degenerate_reasons.append(f"rank_deficient:{rank}/{expected_rank}")
    if not math.isfinite(variance_ratio) or variance_ratio >= READOUT_RESIDUAL_VARIANCE_RATIO_MAX:
        degenerate_reasons.append(
            f"residual_approximately_variance:{variance_ratio:.9g}>={READOUT_RESIDUAL_VARIANCE_RATIO_MAX}"
        )
    before = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    if not degenerate_reasons:
        with torch.no_grad():
            model.params["render_out_w"][:, 3:6].copy_(
                torch.from_numpy(coefficients[:-1].astype(np.float32)).to(
                    model.params["render_out_w"]
                )
            )
            model.params["render_out_b"][3:6].copy_(
                torch.from_numpy(coefficients[-1].astype(np.float32)).to(
                    model.params["render_out_b"]
                )
            )
    after = model.state_dict()
    changed = [name for name in before if not torch.equal(before[name], after[name].detach().cpu())]
    if not degenerate_reasons and set(changed) != {
        "params.render_out_w",
        "params.render_out_b",
    }:
        raise QBT1Error(f"palette init changed tensors outside the permitted QBF1 readout: {changed}")
    if not torch.equal(before["params.render_out_w"][:, :3], after["params.render_out_w"][:, :3].cpu()):
        raise QBT1Error("palette init changed the pose-facing frame-0 RGB readout")
    if not torch.equal(before["params.render_out_b"][:3], after["params.render_out_b"][:3].cpu()):
        raise QBT1Error("palette init changed the pose-facing frame-0 RGB bias")
    receipt = {
        "schema": "ddm_qbt2b_data_dependent_readout_fit.v1",
        "axis": "[macOS-CPU scorer-free data-dependent fit]",
        "score_claim": False,
        "provenance_label": "VIDEO-DERIVED, CE-TRAINED, INHERITED PALETTE; DATA-DEPENDENT READOUT FIT",
        "pair_ids": list(map(int, pair_ids)),
        "selection": "same seeded stratified qbt1/no2 n32",
        "fit_target": "last-frame RGB logit(palette[class]) over observed r1 native-class regions",
        "fit_method": "deterministic pair-by-class evenly-spaced samples; unregularized least squares",
        "lstsq_relative_singular_cutoff": 1.0e-6,
        "sample_count": int(states.shape[0]),
        "feature_count_with_bias": expected_rank,
        "matrix_rank": int(rank),
        "singular_value_max": float(singular_values[0]),
        "singular_value_min": float(singular_values[-1]),
        "condition_number": float(singular_values[0] / singular_values[-1])
        if singular_values[-1] > 0.0
        else math.inf,
        "residual_variance_ratio": variance_ratio,
        "residual_gate_max_exclusive": READOUT_RESIDUAL_VARIANCE_RATIO_MAX,
        "per_class_residual_matrix": class_rows,
        "changed_tensors": changed,
        "frame0_readout_bit_identical": True,
        "qbf1_shapes_unchanged": {
            "render_out_w": list(model.params["render_out_w"].shape),
            "render_out_b": list(model.params["render_out_b"].shape),
        },
        "degenerate": bool(degenerate_reasons),
        "degenerate_reasons": degenerate_reasons,
    }
    samples["target_palette_logit_f64"] = targets
    samples["fitted_palette_logit_f64"] = predicted_logits
    samples["fit_coefficients_f64"] = coefficients
    return receipt, samples


def per_class_argmax_table(
    predicted: np.ndarray, target: np.ndarray
) -> list[dict[str, Any]]:
    predicted = np.asarray(predicted, dtype=np.uint8).reshape(-1)
    target = np.asarray(target, dtype=np.uint8).reshape(-1)
    if predicted.shape != target.shape or predicted.size == 0:
        raise QBT1Error("per-class receipt arrays differ")
    rows = []
    for class_id, class_name in enumerate(PALETTE_CLASSES):
        target_mask = target == class_id
        target_count = int(target_mask.sum())
        predicted_count = int((predicted == class_id).sum())
        within_error = (
            float(np.mean(predicted[target_mask] != class_id)) if target_count else None
        )
        rows.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "target_pixel_share": target_count / predicted.size,
                "predicted_pixel_share": predicted_count / predicted.size,
                "within_class_error": within_error,
                "target_pixels": target_count,
                "predicted_pixels": predicted_count,
                "present_and_below_60pct_error": bool(
                    predicted_count > 0 and within_error is not None and within_error < 0.60
                ),
            }
        )
    return rows


def _load_retained_eval_bank(
    root: Path, pair_ids: Sequence[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    predicted: list[np.ndarray] = []
    target: list[np.ndarray] = []
    poses: list[np.ndarray] = []
    target_poses: list[np.ndarray] = []
    custody: list[dict[str, Any]] = []
    for pair_id in pair_ids:
        path = root / f"pair_{pair_id:04d}.npz"
        fact = file_fact(path)
        with np.load(path, allow_pickle=False) as payload:
            predicted.append(np.asarray(payload["segnet_argmax_u8"], dtype=np.uint8).copy())
            target_key = "target_argmax_u8" if "target_argmax_u8" in payload else "target_argmax"
            pose_key = "posenet_pose6_f32" if "posenet_pose6_f32" in payload else "posenet_pose6"
            target_pose_key = (
                "target_pose6_f32" if "target_pose6_f32" in payload else "target_pose6"
            )
            target.append(np.asarray(payload[target_key], dtype=np.uint8).copy())
            poses.append(np.asarray(payload[pose_key], dtype=np.float32).copy())
            target_poses.append(np.asarray(payload[target_pose_key], dtype=np.float32).copy())
        custody.append({"pair_id": int(pair_id), "payload": fact})
    return (
        np.stack(predicted),
        np.stack(target),
        np.stack(poses),
        np.stack(target_poses),
        custody,
    )


def prepare_inherited_palette_initialization(output: Path) -> dict[str, Any]:
    """Materialize the charter's retained fit and real n32 before/after init receipt."""

    storage = storage_preflight(output, 8 * 1024**3)
    seed_everything(SEED)
    inherited_inputs = verify_qbt2b_inputs()
    palette, fp1_sample_ids = load_fp1_inherited_palette()
    copied_palette = atomic_bytes(output / "palette/inherited_fp1_prototypes_exact.npz", FP1_PALETTE.read_bytes())
    palette_values = atomic_npz(
        output / "palette/inherited_fp1_palette_values.npz",
        palette_rgb_f32=palette,
        fp1_training_sample_ids_i64=fp1_sample_ids,
    )
    model, r1_payload = load_r1_ema_model(torch.device("cpu"))
    raw_samples = collect_readout_fit_samples(model, SELECTION_IDS)
    raw_samples_fact = atomic_npz(output / "fit/readout_fit_samples.npz", **raw_samples)
    fit_receipt, samples = fit_inherited_palette_readout(
        model, palette, SELECTION_IDS, samples=raw_samples
    )
    samples_fact = atomic_npz(output / "fit/readout_fit_samples_and_predictions.npz", **samples)
    fit_receipt["retained_raw_fit_samples"] = raw_samples_fact
    fit_receipt["retained_fit_samples_and_predictions"] = samples_fact
    fit_receipt["source_inputs"] = inherited_inputs
    fit_receipt["palette_exact_copy"] = copied_palette
    fit_receipt["palette_values"] = palette_values
    fit_receipt_fact = atomic_json(output / "fit/READOUT_FIT_RECEIPT.json", fit_receipt)
    if fit_receipt["degenerate"]:
        blocker = {
            "schema": PALETTE_INIT_SCHEMA,
            "status": "BLOCKED_DEGENERATE_READOUT_FIT",
            "disposition": "STOPPED_PER_CHARTER_BEFORE_CE_BUILD",
            "axis": "[macOS-CPU scorer-free data-dependent fit]",
            "score_claim": False,
            "inherited_palette_provenance": "VIDEO-DERIVED, CE-TRAINED, INHERITED",
            "fit_receipt": fit_receipt_fact,
            "storage": storage,
            "training_launched": False,
            "metal_invocations": 0,
            "modal_invocations": 0,
            "contest_eval_invocations": 0,
        }
        atomic_json(output / "INITIALIZATION_RESULT.json", blocker)
        return blocker

    initialized_state = {
        "schema": "ddm_qbt2b_initialized_qbf1_state.v1",
        "source_r1_checkpoint": inherited_inputs["qbt1_r1_checkpoint"],
        "source_r1_ema_updates": int(r1_payload["ema"]["num_updates"]),
        "palette": palette_values,
        "fit_receipt": fit_receipt_fact,
        "state_dict": {
            name: value.detach().cpu().clone() for name, value in model.state_dict().items()
        },
    }
    initialized_state_fact = atomic_torch(output / "initialized/initialized_r3_state.pt", initialized_state)
    params, boundary, interior = model.packet_state()
    qbf1.validate_param_shapes(params)
    if boundary.shape != (N, qbf1.BOUNDARY_LATENT_DIM) or interior.shape != (
        N,
        qbf1.INTERIOR_LATENT_DIM,
    ):
        raise QBT1Error("palette initialized state changed QBF1 latent shapes")

    before_pred, before_target, before_pose, before_target_pose, before_custody = (
        _load_retained_eval_bank(R1_RETAINED, SELECTION_IDS)
    )
    before_table = per_class_argmax_table(before_pred, before_target)
    before_pose_mse = float(
        np.square(before_pose.astype(np.float64) - before_target_pose.astype(np.float64)).mean()
    )

    posenet, segnet = load_differentiable_scorers(REPO / "upstream", device=torch.device("cpu"))
    posenet.eval()
    segnet.eval()
    after_predicted: list[np.ndarray] = []
    after_target: list[np.ndarray] = []
    after_pose: list[np.ndarray] = []
    after_target_pose: list[np.ndarray] = []
    after_rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for chunk_ids in pair_chunks(SELECTION_IDS, 4):
            ids_tensor = torch.tensor(chunk_ids, dtype=torch.long)
            target_argmax, target_pose6 = _target_arrays(chunk_ids, torch.device("cpu"))
            outputs = model(ids_tensor, height=EVAL_H, width=EVAL_W)
            camera = roundtrip_to_camera_uint8_ste(outputs["rgb_pair_01"])
            pose6, logits = scorer_forward(camera, posenet, segnet)
            retained = _retain_eval_outputs(
                output / "init_receipt/after/retained",
                pair_ids=chunk_ids,
                camera=camera,
                pose6=pose6,
                logits=logits,
                target_argmax=target_argmax,
                target_pose6=target_pose6,
            )
            after_rows.extend(retained["rows"])
            after_predicted.append(logits.argmax(dim=1).cpu().numpy().astype(np.uint8))
            after_target.append(target_argmax.cpu().numpy().astype(np.uint8))
            after_pose.append(pose6.cpu().numpy().astype(np.float32))
            after_target_pose.append(target_pose6.cpu().numpy().astype(np.float32))
    after_pred = np.concatenate(after_predicted)
    after_gt = np.concatenate(after_target)
    after_pose_np = np.concatenate(after_pose)
    after_target_pose_np = np.concatenate(after_target_pose)
    after_table = per_class_argmax_table(after_pred, after_gt)
    after_pose_mse = float(
        np.square(after_pose_np.astype(np.float64) - after_target_pose_np.astype(np.float64)).mean()
    )
    gate_count = sum(bool(row["present_and_below_60pct_error"]) for row in after_table)
    init_gate = {
        "threshold": "at least 4 of 5 classes have predicted share > 0 and within-class error < 0.60",
        "passing_class_count": gate_count,
        "passed": gate_count >= 4,
        "miss_does_not_block_ce_birth": True,
    }
    receipt = {
        "schema": PALETTE_INIT_SCHEMA,
        "status": "COMPLETE_NONDEGENERATE_FIT",
        "axis": "[macOS-CPU frozen-scorer advisory]",
        "score_claim": False,
        "selection": "same seeded stratified qbt1/no2 n32",
        "pair_ids": list(SELECTION_IDS),
        "inherited_palette_provenance": "VIDEO-DERIVED, CE-TRAINED, INHERITED",
        "palette_rgb": palette.astype(float).tolist(),
        "palette_custody": {
            "source": inherited_inputs["fp1_palette"],
            "exact_copy": copied_palette,
            "values": palette_values,
            "fp1_training_sample_ids": fp1_sample_ids.astype(int).tolist(),
        },
        "fit_receipt": fit_receipt_fact,
        "initialized_state": initialized_state_fact,
        "qbf1_abi": {
            "immutable": True,
            "render_out_w_shape": list(model.params["render_out_w"].shape),
            "render_out_b_shape": list(model.params["render_out_b"].shape),
            "only_values_changed": True,
        },
        "before": {
            "source": "existing qbt1 r1 retained exact scorer bank",
            "per_class": before_table,
            "pose_mse": before_pose_mse,
            "retained_payloads": before_custody,
        },
        "after": {
            "per_class": after_table,
            "pose_mse": after_pose_mse,
            "retained_payloads": after_rows,
        },
        "init_gate": init_gate,
        "storage": storage,
        "all_receipt_frames_and_scorer_outputs_retained": True,
        "training_launched": False,
        "metal_invocations": 0,
        "modal_invocations": 0,
        "contest_eval_invocations": 0,
    }
    atomic_json(output / "INITIALIZATION_RESULT.json", receipt)
    return receipt


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


# History-slimming law (the r8 O(steps^2) retention wall, memo item 3.2): every
# periodic checkpoint that embeds the full per-step history grows ~2,925 B/step,
# so retained bytes grow quadratically across the checkpoint set.  In sidecar
# mode the append-only journal below is the history's source of truth: periodic
# checkpoints store only (event count, canonical hash, relpath) and stage-end
# checkpoints keep the full embedded history so every existing stage-boundary
# identity check is unchanged.  The journal mirrors the in-memory mutation order
# exactly -- a row event at history.append time, then a patch event for every
# post-append attachment on the SAME row (birth_verdict, checkpoint, reencode)
# -- so replaying the first K events reconstructs the save-time history under
# JSON round-trip, verified fail-closed by the stored canonical hash.
class HistorySidecarJournal:
    def __init__(self, path: Path, *, events: int = 0) -> None:
        self.path = path
        self.events = int(events)

    def append_row(self, row: Mapping[str, Any]) -> None:
        self._write({"kind": "row", "payload": dict(row)})

    def patch_last_row(self, fields: Mapping[str, Any]) -> None:
        self._write({"kind": "patch", "payload": dict(fields)})

    def _write(self, event: Mapping[str, Any]) -> None:
        line = json.dumps(event, sort_keys=True, separators=(",", ":"), default=str)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.events += 1


def replay_history_journal(events: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = []
    for event in events:
        if event.get("kind") == "row":
            history.append(dict(event["payload"]))
        elif event.get("kind") == "patch":
            if not history:
                raise QBT1Error("history sidecar patch precedes any row")
            history[-1].update(event["payload"])
        else:
            raise QBT1Error("history sidecar event kind differs")
    return history


def read_history_journal_events(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise QBT1Error(f"history sidecar journal is absent: {path}")
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    events: list[dict[str, Any]] = []
    for index, line in enumerate(raw_lines):
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            if index == len(raw_lines) - 1:
                break  # a torn final line is the crash signature; earlier corruption refuses
            raise QBT1Error("history sidecar has a corrupt interior line") from None
    return events


def load_history_from_sidecar(
    checkpoint_path: Path, payload: Mapping[str, Any], *, reanchor: bool = False
) -> list[dict[str, Any]]:
    sidecar = (checkpoint_path.parent / str(payload["history_sidecar_relpath"])).resolve()
    events = read_history_journal_events(sidecar)
    journal_events = int(payload["history_journal_events"])
    if len(events) < journal_events:
        raise QBT1Error("history sidecar lost journal events")
    history = replay_history_journal(events[:journal_events])
    if len(history) != int(payload["history_rows"]):
        raise QBT1Error("history sidecar row count differs")
    if canonical_sha256(history) != payload["history_sha256"]:
        raise QBT1Error("history sidecar reconstruction differs")
    if reanchor and len(events) > journal_events:
        # KEEP THE PAYLOAD: retain the pre-reanchor journal (rows past this
        # checkpoint belong to steps the resume will redo), then rewrite the
        # live journal to exactly the replayed prefix so appends continue clean.
        atomic_bytes(
            sidecar.with_name(f"{sidecar.name}.pre_reanchor_step{int(payload['step']):06d}"),
            sidecar.read_bytes(),
        )
        kept = "".join(
            json.dumps(event, sort_keys=True, separators=(",", ":"), default=str) + "\n"
            for event in events[:journal_events]
        )
        atomic_bytes(sidecar, kept.encode("utf-8"))
    return history


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
    curriculum_state: Mapping[str, Any] | None = None,
    history_journal: HistorySidecarJournal | None = None,
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
        "curriculum_state": copy.deepcopy(dict(curriculum_state or {})),
    }
    if history_journal is None:
        payload["history_mode"] = "embedded"
        payload["history"] = list(history)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload["history_mode"] = "sidecar"
        payload["history_journal_events"] = int(history_journal.events)
        payload["history_rows"] = len(history)
        payload["history_sha256"] = canonical_sha256(list(history))
        payload["history_sidecar_relpath"] = os.path.relpath(
            history_journal.path.resolve(), start=path.parent.resolve()
        )
    return atomic_torch(path, payload)


def load_checkpoint(
    path: Path,
    *,
    model: QBFLOWTorch,
    optimizer: torch.optim.Optimizer,
    config: Mapping[str, Any],
    reanchor_sidecar: bool = False,
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
    if payload.get("history_mode", "embedded") == "sidecar":
        history = load_history_from_sidecar(path, payload, reanchor=reanchor_sidecar)
    else:
        history = list(payload["history"])
    return int(payload["step"]), ema, history, payload


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


def _containerize_file_facts(value: object, root: Path, container: Path) -> object:
    if isinstance(value, list):
        return [_containerize_file_facts(item, root, container) for item in value]
    if isinstance(value, dict):
        converted = {
            name: _containerize_file_facts(item, root, container) for name, item in value.items()
        }
        raw_path = converted.get("path")
        if isinstance(raw_path, str):
            path = Path(raw_path)
            try:
                member = path.resolve().relative_to(root.resolve())
            except ValueError:
                return converted
            converted["path"] = f"{container.resolve()}::{member.as_posix()}"
            converted["container"] = str(container.resolve())
            converted["container_member"] = member.as_posix()
        return converted
    return value


def consolidate_reencode_payloads(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Pack one re-encode's retained payloads into one deterministic tar plus one manifest."""

    manifest_path = root / "REENCODE_MANIFEST.json"
    container = root / "reencode_payloads.tar"
    source_files = sorted(
        path for path in root.rglob("*") if path.is_file() and path not in {manifest_path, container}
    )
    if not source_files:
        raise QBT1Error("re-encode consolidation found no retained payloads")
    source_rows = [
        {
            **file_fact(path),
            "relative_path": path.relative_to(root).as_posix(),
        }
        for path in source_files
    ]
    temporary = container.with_name(f".{container.name}.{os.getpid()}.tmp")
    with tarfile.open(temporary, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for path, row in zip(source_files, source_rows, strict=True):
            info = tarfile.TarInfo(row["relative_path"])
            info.size = int(row["bytes"])
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mode = 0o644
            with path.open("rb") as stream:
                archive.addfile(info, stream)
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, container)
    with tarfile.open(container, mode="r") as archive:
        members = {member.name: member for member in archive.getmembers() if member.isfile()}
        if set(members) != {row["relative_path"] for row in source_rows}:
            raise QBT1Error("re-encode tar member set differs before cleanup")
        for row in source_rows:
            extracted = archive.extractfile(members[row["relative_path"]])
            if extracted is None:
                raise QBT1Error("re-encode tar member is unreadable")
            digest = hashlib.sha256(extracted.read()).hexdigest()
            if digest != row["sha256"]:
                raise QBT1Error(f"re-encode tar member sha differs: {row['relative_path']}")
    container_fact = file_fact(container)
    consolidated = _containerize_file_facts(copy.deepcopy(dict(manifest)), root, container)
    assert isinstance(consolidated, dict)
    consolidated.update(
        {
            "retention_mode": "ONE_DETERMINISTIC_TAR_PER_REENCODE",
            "retention_container": container_fact,
            "retention_member_count": len(source_rows),
            "retention_members_logical_bytes": sum(int(row["bytes"]) for row in source_rows),
            "cleanup_certification": {
                "original_root": str(root.resolve()),
                "original_files": source_rows,
                "container": container_fact,
                "reproducibility": "all original bytes and relative paths are SHA-bound in the verified deterministic tar",
                "reason": "avoid measured AP ExFAT 128-KiB cluster amplification",
            },
        }
    )
    for path in sorted(source_files, key=lambda item: len(item.parts), reverse=True):
        path.unlink()
    for directory in sorted(
        (path for path in root.rglob("*") if path.is_dir()),
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        if directory != root and not any(directory.iterdir()):
            directory.rmdir()
    atomic_json(manifest_path, consolidated)
    return consolidated


def reencode_inference_state(
    root: Path,
    *,
    model: QBFLOWTorch,
    state: Mapping[str, torch.Tensor],
    selected_pair_ids: Sequence[int],
    consolidate: bool = False,
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
    return consolidate_reencode_payloads(root, manifest) if consolidate else manifest


def compact_reencode_history(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Keep checkpoint history resumable without recursively embedding coder inventories."""

    compact = {
        "schema": manifest["schema"],
        "archive": copy.deepcopy(manifest["archive"]),
        "B_hat": manifest.get("B_hat"),
        "ht_projection_ready": manifest.get("ht_projection_ready"),
    }
    container = manifest.get("retention_container")
    if container is not None:
        compact["retention_container"] = copy.deepcopy(container)
        manifest_path = Path(container["path"]).parent / "REENCODE_MANIFEST.json"
        compact["manifest"] = file_fact(manifest_path)
    return compact


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
    consolidate: bool = False,
) -> dict[str, Any]:
    """Retain real QBF1 options and a labelled first-order sensitivity table.

    The gradient metric is only a shortlist signal.  This stage deliberately
    does not promote a prequantized option without a realized scorer A/B.
    """

    roles = tuple(sorted({state_tensor_role(name) for name in state}))
    baseline = reencode_inference_state(
        root / "baseline",
        model=model,
        state=state,
        selected_pair_ids=selected_pair_ids,
        consolidate=consolidate,
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
                consolidate=consolidate,
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


# Periodic-checkpoint cadence law: every run keeps >= ~300 periodic saves, so the
# worst-case crash loss is <= ~1/300 of the run (~0.33%; ~126 s at r7's measured
# 2.51 s/step).  The validator bound is max(5, steps // denominator): legacy short
# runs keep the historical every-5 cadence, long runs may coarsen up to the law's
# ceiling -- retention-only, never a treatment variable (saves gate save_checkpoint
# + reencode_inference_state exclusively; verdict cadence is a separate key).
CHECKPOINT_CRASH_LOSS_DENOMINATOR = 300


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


def compile_qbt2b_config(
    *,
    action: str,
    output: Path,
    pair_ids: Sequence[int],
    device: str,
    initialization_state: Path,
    birth_max_steps: int = QBT2B_BIRTH_MAX_STEPS,
    margin_steps: int = QBT2B_MARGIN_STEPS,
    birth_class_weight_mode: str = "none",
    birth_event_mode: str = "accuracy_020",
    margin_constraint_mode: str = MARGIN_CONSTRAINT_UNCONSTRAINED,
    checkpoint_every_steps: int | None = None,
    checkpoint_history_mode: str = "embedded",
) -> dict[str, Any]:
    if birth_event_mode not in BIRTH_EVENT_MODE_THRESHOLDS:
        raise QBT1Error(f"unsupported QBT2B birth event mode: {birth_event_mode}")
    if margin_constraint_mode not in MARGIN_CONSTRAINT_MODE_PINS:
        raise QBT1Error(f"unsupported QBT2B margin constraint mode: {margin_constraint_mode}")
    total_steps = int(birth_max_steps) + int(margin_steps)
    config = compile_config(
        action=action,
        output=output,
        pair_ids=pair_ids,
        steps=total_steps,
        device=device,
    )
    if checkpoint_every_steps is not None:
        config["checkpoint_every_steps"] = int(checkpoint_every_steps)
    if checkpoint_history_mode != "embedded":
        config["checkpoint_history_mode"] = str(checkpoint_history_mode)
    initialization = file_fact(initialization_state)
    margin_constraint_pin = MARGIN_CONSTRAINT_MODE_PINS[margin_constraint_mode]
    config.update(
        {
            "curriculum_mode": "ce_birth_then_margin",
            "birth_max_steps": int(birth_max_steps),
            "margin_steps": int(margin_steps),
            "birth_verdict_every_steps": max(1, min(5, int(birth_max_steps))),
            "birth_stability_verdicts": 2,
            "birth_within_class_error_max": BIRTH_EVENT_MODE_THRESHOLDS[birth_event_mode],
            "birth_event_mode": str(birth_event_mode),
            "birth_class_weight_mode": str(birth_class_weight_mode),
            "margin_constraint_mode": str(margin_constraint_mode),
            "margin_constraint_bounds": copy.deepcopy(margin_constraint_pin["bounds"]),
            "margin_constraint_eta_lambda": float(margin_constraint_pin["eta_lambda"]),
            "initialization_state_path": initialization["path"],
            "initialization_state_sha256": initialization["sha256"],
            "consolidate_checkpoint_reencodes": True,
        }
    )
    config["ema"] = resolve_ema_law(total_steps)
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
        "curriculum_mode", "birth_max_steps", "margin_steps", "birth_verdict_every_steps",
        "birth_stability_verdicts", "birth_within_class_error_max", "initialization_state_path",
        "initialization_state_sha256", "consolidate_checkpoint_reencodes", "birth_class_weight_mode",
        "birth_event_mode",
        "margin_constraint_mode", "margin_constraint_bounds", "margin_constraint_eta_lambda",
        "checkpoint_history_mode",
    }
    if unknown:
        raise QBT1Error(f"compiled config has unknown fields: {sorted(unknown)}")
    if str(config.get("checkpoint_history_mode", "embedded")) not in {"embedded", "sidecar"}:
        raise QBT1Error("checkpoint history mode differs")
    if int(config["num_pairs"]) != N or int(config["chunk_pairs_hard_ceiling"]) != MAX_CHUNK_PAIRS:
        raise QBT1Error("frozen population or chunk ceiling differs")
    ids = tuple(map(int, config["pair_ids"]))
    if not ids or len(ids) != len(set(ids)) or min(ids) < 0 or max(ids) >= N:
        raise QBT1Error("compiled pair IDs are not a unique subset of n600")
    if not 1 <= int(config["chunk_pairs"]) <= MAX_CHUNK_PAIRS:
        raise QBT1Error("chunk_pairs exceeds the hard-coded ceiling of 30")
    if config["retain_all_payloads"] is not True or int(config["pose_start_step"]) != 0:
        raise QBT1Error("payload retention or step-zero pose binding differs")
    checkpoint_cadence_ceiling = max(5, int(config["steps"]) // CHECKPOINT_CRASH_LOSS_DENOMINATOR)
    if (
        int(config["seed"]) != SEED
        or int(config["minimum_free_bytes"]) < 8 * 1024**3
        or not 1 <= int(config["checkpoint_every_steps"]) <= checkpoint_cadence_ceiling
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
    curriculum_mode = str(config.get("curriculum_mode", "legacy_margin_only"))
    if str(config.get("birth_class_weight_mode", "none")) not in ("none", "balanced"):
        raise QBT1Error("QBT2B birth class-weight mode differs")
    birth_event_mode = str(config.get("birth_event_mode", "accuracy_020"))
    if birth_event_mode not in BIRTH_EVENT_MODE_THRESHOLDS:
        raise QBT1Error("QBT2B birth event mode differs")
    margin_constraint_mode = str(
        config.get("margin_constraint_mode", MARGIN_CONSTRAINT_UNCONSTRAINED)
    )
    if margin_constraint_mode not in MARGIN_CONSTRAINT_MODE_PINS:
        raise QBT1Error("QBT2B margin constraint mode differs")
    margin_constraint_pin = MARGIN_CONSTRAINT_MODE_PINS[margin_constraint_mode]
    margin_constraint_bounds = config.get("margin_constraint_bounds", {})
    margin_constraint_eta = float(config.get("margin_constraint_eta_lambda", 0.0))
    expected_bounds = margin_constraint_pin["bounds"]
    if not isinstance(margin_constraint_bounds, Mapping) or set(margin_constraint_bounds) != set(
        expected_bounds
    ):
        raise QBT1Error("QBT2B margin constraint mode/bounds/eta group differs")
    if any(
        not math.isclose(
            float(margin_constraint_bounds[name]),
            float(expected_bounds[name]),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        for name in expected_bounds
    ) or not math.isclose(
        margin_constraint_eta,
        float(margin_constraint_pin["eta_lambda"]),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise QBT1Error("QBT2B margin constraint mode/bounds/eta group differs")
    if curriculum_mode == "ce_birth_then_margin":
        required = {
            "birth_max_steps",
            "margin_steps",
            "birth_verdict_every_steps",
            "birth_stability_verdicts",
            "birth_within_class_error_max",
            "initialization_state_path",
            "initialization_state_sha256",
            "consolidate_checkpoint_reencodes",
        }
        missing = required - set(config)
        if missing:
            raise QBT1Error(f"QBT2B config lacks additive fields: {sorted(missing)}")
        if int(config["steps"]) != int(config["birth_max_steps"]) + int(config["margin_steps"]):
            raise QBT1Error("QBT2B total step geometry differs")
        if (
            int(config["birth_max_steps"]) < 1
            or int(config["margin_steps"]) < 1
            or not 1 <= int(config["birth_verdict_every_steps"]) <= int(config["birth_max_steps"])
            or int(config["birth_stability_verdicts"]) != 2
            or not math.isclose(
                float(config["birth_within_class_error_max"]),
                BIRTH_EVENT_MODE_THRESHOLDS[birth_event_mode],
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            or config["consolidate_checkpoint_reencodes"] is not True
        ):
            raise QBT1Error("QBT2B CE-birth/event/retention law differs")
        initialization = file_fact(Path(config["initialization_state_path"]))
        if initialization["sha256"] != config["initialization_state_sha256"]:
            raise QBT1Error("QBT2B initialized state custody differs")
    elif curriculum_mode != "legacy_margin_only":
        raise QBT1Error(f"unsupported curriculum mode: {curriculum_mode}")


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
    if config.get("curriculum_mode", "legacy_margin_only") == "ce_birth_then_margin":
        initialization_path = Path(config["initialization_state_path"])
        if sha256_file(initialization_path) != config["initialization_state_sha256"]:
            raise QBT1Error("QBT2B initialized state drifted before model construction")
        initialization = torch.load(initialization_path, map_location="cpu", weights_only=False)
        if initialization.get("schema") != "ddm_qbt2b_initialized_qbf1_state.v1":
            raise QBT1Error("QBT2B initialized state schema differs")
        model = load_initial_model(device)
        model.load_state_dict(
            {
                name: value.detach().clone().to(device)
                for name, value in initialization["state_dict"].items()
            },
            strict=True,
        )
    else:
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
    curriculum_mode = str(config.get("curriculum_mode", "legacy_margin_only"))
    consolidate_reencodes = bool(config.get("consolidate_checkpoint_reencodes", False))
    curriculum_state: dict[str, Any] = {
        "phase": "stage_03_joint_boundary_interior_birth",
        "birth_step": 0,
        "margin_step": 0,
        "birth_stable_verdicts": 0,
        "birth_verdict_count": 0,
        "birth_handoff_authorized": curriculum_mode == "legacy_margin_only",
    }
    if config.get("resume_from"):
        step, ema, history, payload = load_checkpoint(
            Path(config["resume_from"]),
            model=model,
            optimizer=optimizer,
            config=config,
            reanchor_sidecar=True,
        )
        curriculum_state.update(payload.get("curriculum_state", {}))
        resume_identity = {
            "checkpoint": file_fact(Path(config["resume_from"])),
            "live_state_sha256": canonical_sha256(
                {name: hashlib.sha256(value.numpy().tobytes()).hexdigest() for name, value in payload["live_state_dict"].items()}
            ),
            "ema_state_sha256": canonical_sha256(
                {name: hashlib.sha256(value.numpy().tobytes()).hexdigest() for name, value in payload["ema"]["shadow"].items()}
            ),
        }

    history_mode = str(config.get("checkpoint_history_mode", "embedded"))
    history_journal = None
    if history_mode == "sidecar":
        journal_path = output / "history_journal.jsonl"
        if config.get("resume_from"):
            if payload.get("history_mode", "embedded") != "sidecar":
                raise QBT1Error("resume history mode differs")
            history_journal = HistorySidecarJournal(
                journal_path, events=int(payload["history_journal_events"])
            )
        else:
            atomic_bytes(journal_path, b"")
            history_journal = HistorySidecarJournal(journal_path, events=0)
    elif config.get("resume_from") and payload.get("history_mode", "embedded") != "embedded":
        raise QBT1Error("resume history mode differs")

    margin_constraint_mode = str(
        config.get("margin_constraint_mode", MARGIN_CONSTRAINT_UNCONSTRAINED)
    )
    margin_constraint_enabled = margin_constraint_mode == MARGIN_CONSTRAINT_LANE_MOVABLE
    margin_constraint_bounds = {
        name: float(value) for name, value in config.get("margin_constraint_bounds", {}).items()
    }
    margin_constraint_eta = float(config.get("margin_constraint_eta_lambda", 0.0))
    if margin_constraint_enabled:
        stored_constraint_state = curriculum_state.get("margin_constraint_state")
        if stored_constraint_state is None:
            stored_constraint_state = {
                "mode": margin_constraint_mode,
                "lambdas": dict.fromkeys(margin_constraint_bounds, 0.0),
            }
            curriculum_state["margin_constraint_state"] = stored_constraint_state
        if (
            stored_constraint_state.get("mode") != margin_constraint_mode
            or set(stored_constraint_state.get("lambdas", {})) != set(margin_constraint_bounds)
        ):
            raise QBT1Error("resumed margin-constraint state differs")
        for value in stored_constraint_state["lambdas"].values():
            if not 0.0 <= float(value) <= MARGIN_CONSTRAINT_LAMBDA_MAX:
                raise QBT1Error("resumed margin-constraint lambda differs")

    stage03a_checkpoint = None
    stage03a_reencode = None
    boundary_resume_identity = None
    if curriculum_mode == "ce_birth_then_margin" and not curriculum_state.get(
        "birth_handoff_authorized", False
    ):
        birth_max_steps = int(config["birth_max_steps"])
        birth_class_weight_mode = str(config.get("birth_class_weight_mode", "none"))
        birth_class_weights = None
        if birth_class_weight_mode == "balanced":
            birth_class_weights = derive_balanced_class_weights(pair_ids, device)
            curriculum_state["birth_class_weights"] = [
                float(value) for value in birth_class_weights.detach().cpu()
            ]
        for birth_index in range(int(curriculum_state["birth_step"]), birth_max_steps):
            chunk_ids = optimizer_chunks[birth_index % len(optimizer_chunks)]
            ids_tensor = torch.tensor(chunk_ids, dtype=torch.long, device=device)
            target_argmax, target_pose6 = _target_arrays(chunk_ids, device)
            sample_weights = no2_sample_weights(chunk_ids, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(ids_tensor, height=EVAL_H, width=EVAL_W)
            camera = roundtrip_to_camera_uint8_ste(outputs["rgb_pair_01"])
            pose6, logits = scorer_forward(camera, posenet, segnet)
            total, components = realized_ce_birth_objective(
                camera,
                pose6,
                logits,
                target_argmax,
                target_pose6,
                sample_weights,
                class_weights=birth_class_weights,
            )
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ema.update(model)
            step += 1
            curriculum_state["birth_step"] = birth_index + 1
            row = {
                "step": step,
                "stage": "stage_03a_ce_class_birth",
                "stage_step": birth_index + 1,
                "pair_ids": list(chunk_ids),
                "materialized_pairs": len(chunk_ids),
                "objective": {
                    name: float(value.detach().cpu()) for name, value in components.items()
                },
                "ema_effective_decay": ema.effective_decay(),
            }
            history.append(row)
            if history_journal is not None:
                history_journal.append_row(row)
            checkpoint_due = (birth_index + 1) % int(config["checkpoint_every_steps"]) == 0
            verdict_due = (birth_index + 1) % int(config["birth_verdict_every_steps"]) == 0
            handoff_now = False
            if verdict_due or birth_index + 1 == birth_max_steps:
                verdict_index = int(curriculum_state["birth_verdict_count"]) + 1
                verdict = evaluate_birth_verdict(
                    output / "stage_03a_ce_class_birth/verdicts",
                    model=model,
                    ema=ema,
                    posenet=posenet,
                    segnet=segnet,
                    pair_ids=pair_ids,
                    chunk_pairs=int(config["chunk_pairs"]),
                    step=step,
                    verdict_index=verdict_index,
                    within_class_error_max=float(config["birth_within_class_error_max"]),
                )
                curriculum_state["birth_verdict_count"] = verdict_index
                if verdict["gate"]["all_five_classes_pass"]:
                    curriculum_state["birth_stable_verdicts"] = int(
                        curriculum_state["birth_stable_verdicts"]
                    ) + 1
                else:
                    curriculum_state["birth_stable_verdicts"] = 0
                verdict["consecutive_passing_verdicts"] = int(
                    curriculum_state["birth_stable_verdicts"]
                )
                verdict["required_consecutive_passing_verdicts"] = int(
                    config["birth_stability_verdicts"]
                )
                row["birth_verdict"] = verdict
                if history_journal is not None:
                    history_journal.patch_last_row({"birth_verdict": verdict})
                if int(curriculum_state["birth_stable_verdicts"]) >= int(
                    config["birth_stability_verdicts"]
                ):
                    curriculum_state["birth_handoff_authorized"] = True
                    curriculum_state["phase"] = "stage_03_joint_boundary_interior_birth"
                    handoff_now = True
            if checkpoint_due or verdict_due or birth_index + 1 == birth_max_steps:
                checkpoint = save_checkpoint(
                    output
                    / "stage_03a_ce_class_birth/checkpoints"
                    / f"periodic_step_{birth_index + 1:06d}.pt",
                    model=model,
                    optimizer=optimizer,
                    ema=ema,
                    config=config,
                    step=step,
                    stage="stage_03a_ce_class_birth",
                    history=history,
                    curriculum_state=curriculum_state,
                    history_journal=history_journal,
                )
                checkpoint_reencode = reencode_inference_state(
                    output
                    / "stage_03a_ce_class_birth/reencoded"
                    / f"step_{birth_index + 1:06d}",
                    model=model,
                    state=ema.shadow,
                    selected_pair_ids=pair_ids,
                    consolidate=consolidate_reencodes,
                )
                row["checkpoint"] = checkpoint
                row["reencode"] = compact_reencode_history(checkpoint_reencode)
                if history_journal is not None:
                    history_journal.patch_last_row(
                        {"checkpoint": row["checkpoint"], "reencode": row["reencode"]}
                    )
            if handoff_now:
                break

        stage03a_checkpoint = save_checkpoint(
            output
            / "stage_03a_ce_class_birth/checkpoints"
            / (
                "stage_03a_end.pt"
                if curriculum_state["birth_handoff_authorized"]
                else "stage_03a_cap_without_handoff.pt"
            ),
            model=model,
            optimizer=optimizer,
            ema=ema,
            config=config,
            step=step,
            stage=(
                "stage_03a_ce_class_birth_end"
                if curriculum_state["birth_handoff_authorized"]
                else "stage_03a_ce_class_birth_cap_without_handoff"
            ),
            history=history,
            curriculum_state=curriculum_state,
        )
        stage03a_reencode = reencode_inference_state(
            output
            / "stage_03a_ce_class_birth/reencoded"
            / (
                "stage_03a_end"
                if curriculum_state["birth_handoff_authorized"]
                else "stage_03a_cap_without_handoff"
            ),
            model=model,
            state=ema.shadow,
            selected_pair_ids=pair_ids,
            consolidate=consolidate_reencodes,
        )
        boundary_model, boundary_optimizer, _unused_boundary_ema = _model_and_optimizer(config, device)
        boundary_step, boundary_ema, boundary_history, boundary_payload = load_checkpoint(
            Path(stage03a_checkpoint["path"]),
            model=boundary_model,
            optimizer=boundary_optimizer,
            config=config,
        )
        live_identical = all(
            torch.equal(value.detach().cpu(), boundary_model.state_dict()[name].detach().cpu())
            for name, value in model.state_dict().items()
        )
        ema_identical = all(
            torch.equal(value.detach().cpu(), boundary_ema.shadow[name].detach().cpu())
            for name, value in ema.shadow.items()
        )
        boundary_resume_identity = {
            "bit_faithful": bool(
                boundary_step == step
                and boundary_history == history
                and live_identical
                and ema_identical
                and boundary_payload.get("curriculum_state") == curriculum_state
            ),
            "checkpoint": stage03a_checkpoint,
            "archive_before_reload": stage03a_reencode["archive"],
            "rng_restored": boundary_payload.get("rng") is not None,
            "handoff_authorized_by_realized_event": bool(
                curriculum_state["birth_handoff_authorized"]
            ),
        }
        boundary_reencode = reencode_inference_state(
            output / "stage_03a_ce_class_birth/reencoded/resume_identity",
            model=boundary_model,
            state=boundary_ema.shadow,
            selected_pair_ids=pair_ids,
            consolidate=consolidate_reencodes,
        )
        boundary_resume_identity["archive_after_reload"] = boundary_reencode["archive"]
        boundary_resume_identity["bit_faithful"] = bool(
            boundary_resume_identity["bit_faithful"]
            and boundary_reencode["archive"]["sha256"]
            == stage03a_reencode["archive"]["sha256"]
        )
        if not boundary_resume_identity["bit_faithful"]:
            raise QBT1Error("resume identity across the 03a/03 boundary differs")
        model, optimizer, ema = boundary_model, boundary_optimizer, boundary_ema
        if not curriculum_state["birth_handoff_authorized"]:
            peak_rss = _maximum_rss_bytes()
            memory = project_memory(
                baseline_rss_bytes=baseline_rss,
                observed_peak_rss_bytes=peak_rss,
                observed_pairs=max(map(len, chunks)),
                real_chunk_pairs=REAL_TRAIN_CHUNK_PAIRS,
            )
            result = {
                "schema": RESULT_SCHEMA,
                "complete": True,
                "arm": "ddm_qbt2b_ce_birth_stage_smoke_or_cap",
                "status": "BIRTH_STAGE_CAPPED_WITHOUT_EVENT_HANDOFF",
                "axis": "[macOS-CPU bounded mechanism smoke; not a verdict]"
                if config["action"] == "smoke"
                else "[macOS-MPS governed n32 research row; not contest authority]",
                "score_claim": False,
                "promotion_eligible": False,
                "pointer_moved": False,
                "pins": pins,
                "storage": storage,
                "history": history,
                "stage_03a_checkpoint": stage03a_checkpoint,
                "stage_03a_reencode": stage03a_reencode,
                "resume_identity_across_03a_03": boundary_resume_identity,
                "memory_projection": memory,
                "all_payloads_retained": True,
                "training_launch": config["action"] == "train",
                "metal_invocations": int(config["action"] == "train"),
                "modal_invocations": 0,
                "contest_eval_invocations": 0,
                "boundary": "margin stage did not run because the realized five-class event was not stable twice",
            }
            atomic_json(output / "RESULT.json", result)
            return result

    if curriculum_mode == "ce_birth_then_margin":
        margin_start = int(curriculum_state["margin_step"])
        margin_steps = int(config["margin_steps"])
    else:
        margin_start = step
        margin_steps = int(config["steps"])

    for current in range(margin_start, margin_steps):
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
            margin_steps,
            float(config["expected_flip_tau_start"]),
            float(config["expected_flip_tau_end"]),
        )
        margin_constraint_telemetry = None
        margin_constraint_lambdas = None
        if margin_constraint_enabled:
            margin_constraint_state = curriculum_state["margin_constraint_state"]
            lambda_before = {
                name: float(value)
                for name, value in margin_constraint_state["lambdas"].items()
            }
            realized_werr = {
                "Lane": realized_within_class_error(logits, target_argmax, 1),
                "Movable": realized_within_class_error(logits, target_argmax, 3),
            }
            margin_constraint_lambdas = dual_ascent_margin_constraints(
                lambda_before,
                realized_werr,
                margin_constraint_bounds,
                eta_lambda=margin_constraint_eta,
            )
            margin_constraint_state["lambdas"] = copy.deepcopy(margin_constraint_lambdas)
            margin_constraint_telemetry = {
                "mode": margin_constraint_mode,
                "bounds": copy.deepcopy(margin_constraint_bounds),
                "eta_lambda": margin_constraint_eta,
                "lambda_max": MARGIN_CONSTRAINT_LAMBDA_MAX,
                "realized_within_class_error": realized_werr,
                "constraint_residual": {
                    name: realized_werr[name] - margin_constraint_bounds[name]
                    for name in margin_constraint_bounds
                },
                "binding": {
                    name: realized_werr[name] > margin_constraint_bounds[name]
                    for name in margin_constraint_bounds
                },
                "lambda_before": lambda_before,
                "lambda_after": copy.deepcopy(margin_constraint_lambdas),
                "lambda_at_ceiling": {
                    name: math.isclose(
                        value,
                        MARGIN_CONSTRAINT_LAMBDA_MAX,
                        rel_tol=0.0,
                        abs_tol=1.0e-12,
                    )
                    for name, value in margin_constraint_lambdas.items()
                },
                "realization_path": "render->R->uint8->frozen_SegNet_argmax",
            }
        total, components = joint_objective(
            outputs,
            camera,
            pose6,
            logits,
            target_argmax,
            target_pose6,
            tau,
            sample_weights,
            margin_constraint_lambdas,
        )
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        ema.update(model)
        step += 1
        curriculum_state["margin_step"] = current + 1
        row = {
            "step": step,
            "stage": "stage_03_joint_boundary_interior_birth",
            "stage_step": current + 1,
            "pair_ids": list(chunk_ids),
            "materialized_pairs": len(chunk_ids),
            "objective": {name: float(value.detach().cpu()) for name, value in components.items()},
            "ema_effective_decay": ema.effective_decay(),
        }
        if margin_constraint_telemetry is not None:
            row["margin_constraint"] = margin_constraint_telemetry
        history.append(row)
        if history_journal is not None:
            history_journal.append_row(row)
        checkpoint_due = (current + 1) % int(config["checkpoint_every_steps"]) == 0 or current + 1 == margin_steps
        if checkpoint_due:
            checkpoint = save_checkpoint(
                output / "stage_03_joint_boundary_interior_birth/checkpoints" / f"periodic_step_{current + 1:06d}.pt",
                model=model,
                optimizer=optimizer,
                ema=ema,
                config=config,
                step=step,
                stage="stage_03_joint_boundary_interior_birth",
                history=history,
                curriculum_state=curriculum_state,
                history_journal=history_journal,
            )
            checkpoint_reencode = reencode_inference_state(
                output / "stage_03_joint_boundary_interior_birth/reencoded" / f"step_{current + 1:06d}",
                model=model,
                state=ema.shadow,
                selected_pair_ids=pair_ids,
                consolidate=consolidate_reencodes,
            )
            row["checkpoint"] = checkpoint
            row["reencode"] = (
                compact_reencode_history(checkpoint_reencode)
                if curriculum_mode == "ce_birth_then_margin"
                else checkpoint_reencode
            )
            if history_journal is not None:
                history_journal.patch_last_row(
                    {"checkpoint": row["checkpoint"], "reencode": row["reencode"]}
                )

    stage3_checkpoint = save_checkpoint(
        output / "stage_03_joint_boundary_interior_birth/checkpoints/stage_03_end.pt",
        model=model,
        optimizer=optimizer,
        ema=ema,
        config=config,
        step=step,
        stage="stage_03_joint_boundary_interior_birth_end",
        history=history,
        curriculum_state=curriculum_state,
    )
    stage3_reencode = reencode_inference_state(
        output / "stage_03_joint_boundary_interior_birth/reencoded/stage_03_end",
        model=model,
        state=ema.shadow,
        selected_pair_ids=pair_ids,
        consolidate=consolidate_reencodes,
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
    if (
        resumed_step != step
        or resumed_history != history
        or resumed_payload.get("curriculum_state") != curriculum_state
    ):
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
        consolidate=consolidate_reencodes,
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
        consolidate=consolidate_reencodes,
    )
    stage4_checkpoint = save_checkpoint(
        output / "stage_04_precision_waterfill_and_byteclose/checkpoints/stage_04_end.pt",
        model=resumed_model,
        optimizer=resumed_optimizer,
        ema=resumed_ema,
        config=config,
        step=step,
        stage="stage_04_precision_waterfill_and_byteclose_end",
        history=history,
        curriculum_state=curriculum_state,
    )
    stage4_reencode = reencode_inference_state(
        output / "stage_04_precision_waterfill_and_byteclose/reencoded/stage_04_end",
        model=resumed_model,
        state=resumed_ema.shadow,
        selected_pair_ids=pair_ids,
        consolidate=consolidate_reencodes,
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
        step=step,
        stage="stage_05_same_budget_admission_end",
        history=history,
        curriculum_state=curriculum_state,
    )
    stage5_reencode = reencode_inference_state(
        output / "stage_05_same_budget_admission/reencoded/stage_05_end",
        model=resumed_model,
        state=resumed_ema.shadow,
        selected_pair_ids=pair_ids,
        consolidate=consolidate_reencodes,
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
        "curriculum_mode": curriculum_mode,
        "stage_03a_checkpoint": stage03a_checkpoint,
        "stage_03a_reencode": stage03a_reencode,
        "resume_identity_across_03a_03": boundary_resume_identity,
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


def latest_birth_verdict(history: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    rows = [row for row in history if "birth_verdict" in row]
    if not rows:
        raise QBT1Error("storage projection smoke lacks a retained birth verdict row")
    return rows[-1]["birth_verdict"]


def latest_birth_verdict_pair_ids(history: Sequence[Mapping[str, Any]]) -> tuple[int, ...]:
    return tuple(map(int, latest_birth_verdict(history)["pair_ids"]))


def project_on_disk_storage(
    smoke_result: Mapping[str, Any],
    *,
    output: Path,
    total_steps: int,
    checkpoint_every_steps: int,
    birth_max_steps: int,
    birth_verdict_every_steps: int,
    history_mode: str = "embedded",
) -> dict[str, Any]:
    """Project AP on-disk allocation using the charter's cluster-aware formula."""

    output.mkdir(parents=True, exist_ok=True)
    fs = os.statvfs(output)
    cluster_size = int(fs.f_frsize)
    live_free = int(fs.f_bavail * fs.f_frsize)
    r2_first_checkpoint = file_fact(R2_FIRST_CHECKPOINT)
    r2_final_checkpoint = file_fact(R2_FINAL_CHECKPOINT)
    r2_history_growth_per_step = (
        int(r2_final_checkpoint["bytes"]) - int(r2_first_checkpoint["bytes"])
    ) / (4865 - 5)
    projected_final_checkpoint_bytes = math.ceil(
        int(r2_first_checkpoint["bytes"]) + r2_history_growth_per_step * (int(total_steps) - 5)
    )
    base_checkpoint_bytes = max(
        int(smoke_result["stage_03a_checkpoint"]["bytes"]),
        R1_CHECKPOINT.stat().st_size,
    )
    checkpoint_bytes = max(base_checkpoint_bytes, projected_final_checkpoint_bytes)
    # Sidecar mode breaks the O(steps^2) wall: periodic checkpoints carry no
    # embedded history (base size only) and the history's bytes are paid ONCE in
    # the append-only journal (2x safety factor for JSON-vs-pickle row size,
    # doubled again for one retained pre-reanchor copy).  Stage-end checkpoints
    # still embed the full history, so checkpoint_unit keeps the worst case.
    periodic_checkpoint_bytes = (
        base_checkpoint_bytes if history_mode == "sidecar" else checkpoint_bytes
    )
    sidecar_projection = 0
    if history_mode == "sidecar":
        sidecar_logical = math.ceil(2 * r2_history_growth_per_step * int(total_steps))
        sidecar_projection = 2 * (sidecar_logical + 4096 + 2 * cluster_size)
    smoke_reencode = smoke_result["stage_03a_reencode"]
    smoke_manifest = file_fact(
        Path(smoke_reencode["retention_container"]["path"]).parent / "REENCODE_MANIFEST.json"
    )
    repack_reference = file_fact(R2_REPACK_REFERENCE_TAR)
    smoke_birth_pair_ids = latest_birth_verdict_pair_ids(smoke_result["history"])
    theoretical_smoke_members = 36 + 4 + 1 + 8 * len(
        smoke_birth_pair_ids
    )
    theoretical_full_members = 36 + 4 + 1 + 8 * len(SELECTION_IDS)
    full_member_count = math.ceil(
        int(smoke_reencode["retention_member_count"])
        * theoretical_full_members
        / theoretical_smoke_members
    )
    projected_manifest_bytes = math.ceil(
        int(smoke_manifest["bytes"])
        * full_member_count
        / int(smoke_reencode["retention_member_count"])
    )
    projected_tar_bytes = max(
        int(smoke_reencode["retention_container"]["bytes"]), int(repack_reference["bytes"])
    )
    # AP is ExFAT under macOS: each ordinary file has a measured 4-KiB AppleDouble
    # companion, and every file consumes one 128-KiB allocation cluster.  A periodic
    # unit is checkpoint + tar + manifest plus their three companions => six files.
    files_per_periodic_unit = 6
    per_checkpoint_logical_bytes = (
        periodic_checkpoint_bytes + projected_tar_bytes + projected_manifest_bytes + 3 * 4096
    )
    periodic_count = math.ceil(int(total_steps) / int(checkpoint_every_steps))
    periodic_projection = (
        per_checkpoint_logical_bytes + files_per_periodic_unit * cluster_size
    ) * periodic_count

    smoke_birth_verdict = latest_birth_verdict(smoke_result["history"])
    smoke_verdict = smoke_birth_verdict["retained_payload"]
    smoke_pairs = len(smoke_birth_verdict["pair_ids"])
    verdict_logical = math.ceil(int(smoke_verdict["bytes"]) * len(SELECTION_IDS) / smoke_pairs)
    verdict_count = math.ceil(int(birth_max_steps) / int(birth_verdict_every_steps))
    verdict_projection = (verdict_logical + 2 * 4096 + 2 * cluster_size) * verdict_count

    reencode_unit = projected_tar_bytes + projected_manifest_bytes + 2 * 4096 + 4 * cluster_size
    checkpoint_unit = checkpoint_bytes + 4096 + 2 * cluster_size
    # stage03a end; stage03 end; stage04 resume; 8 roles x 4 bits + baseline;
    # stage04 end; stage05 end = 38 additional re-encodes.  Four stage-end
    # checkpoints are additional to periodic checkpoints.
    extra_reencode_count = 38
    extra_checkpoint_count = 4
    stage_boundary_projection = (
        extra_reencode_count * reencode_unit + extra_checkpoint_count * checkpoint_unit
    )
    final_eval_projection = verdict_logical + 2 * cluster_size
    fixed_metadata_reserve = 128 * 1024**2
    projected_bytes = (
        periodic_projection
        + verdict_projection
        + stage_boundary_projection
        + final_eval_projection
        + sidecar_projection
        + fixed_metadata_reserve
    )
    safety_reserve = math.ceil(projected_bytes * 0.10)
    required_post_run_free = 8 * 1024**3
    required_live_free = projected_bytes + safety_reserve + required_post_run_free
    return {
        "schema": "ddm_qbt2b_cluster_aware_storage_projection.v1",
        "axis": "[macOS APDataStore on-disk projection; no score claim]",
        "score_claim": False,
        "filesystem": {
            "path": str(output.resolve()),
            "cluster_size_bytes": cluster_size,
            "live_free_bytes": live_free,
            "measured_appledouble_bytes_per_file": 4096,
        },
        "real_schedule": {
            "birth_max_steps": int(birth_max_steps),
            "margin_steps": int(total_steps) - int(birth_max_steps),
            "total_steps": int(total_steps),
            "checkpoint_every_steps": int(checkpoint_every_steps),
            "periodic_checkpoint_count": periodic_count,
            "birth_verdict_every_steps": int(birth_verdict_every_steps),
            "birth_verdict_count_max": verdict_count,
        },
        "measured_inputs": {
            "smoke_checkpoint": smoke_result["stage_03a_checkpoint"],
            "r1_checkpoint": file_fact(R1_CHECKPOINT),
            "r2_first_checkpoint": r2_first_checkpoint,
            "r2_final_checkpoint": r2_final_checkpoint,
            "r2_measured_history_growth_bytes_per_step": r2_history_growth_per_step,
            "projected_worst_case_final_checkpoint_bytes": projected_final_checkpoint_bytes,
            "smoke_reencode_tar": smoke_reencode["retention_container"],
            "smoke_reencode_manifest": smoke_manifest,
            "r2_repacked_n32_tar": repack_reference,
            "smoke_verdict_payload": smoke_verdict,
        },
        "checkpoint_formula": {
            "literal": "(per_checkpoint_logical_bytes + n_files * fs_cluster_size) * ceil(total_steps / checkpoint_every_steps)",
            "per_checkpoint_logical_bytes": per_checkpoint_logical_bytes,
            "checkpoint_component_uses_worst_case_final_size_for_every_period": history_mode != "sidecar",
            "history_mode": history_mode,
            "n_files": files_per_periodic_unit,
            "fs_cluster_size": cluster_size,
            "checkpoint_count": periodic_count,
            "projected_bytes": periodic_projection,
        },
        "other_retained_demand": {
            "birth_verdicts_bytes": verdict_projection,
            "stage_boundary_and_precision_bytes": stage_boundary_projection,
            "final_n32_evaluation_bytes": final_eval_projection,
            "history_sidecar_bytes": sidecar_projection,
            "fixed_metadata_reserve_bytes": fixed_metadata_reserve,
        },
        "projected_bytes": projected_bytes,
        "ten_percent_safety_reserve_bytes": safety_reserve,
        "required_post_run_free_bytes": required_post_run_free,
        "required_live_free_bytes": required_live_free,
        "live_headroom_after_requirement_bytes": live_free - required_live_free,
        "passes_live_df": live_free >= required_live_free,
        "reencode_retention_default": "one deterministic tar plus one manifest per re-encode",
        "fail_closed": True,
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


def compiled_qbt2b_launch_request(
    *,
    initialization_result_path: Path,
    smoke_result_path: Path,
    review_receipt_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    initialization = json.loads(initialization_result_path.read_text(encoding="utf-8"))
    smoke = json.loads(smoke_result_path.read_text(encoding="utf-8"))
    review = json.loads(review_receipt_path.read_text(encoding="utf-8"))
    expected_review_files = {
        "experiments/ddm_qbt1_qbflow_trainer.py",
        "experiments/tests/test_ddm_qbt1_qbflow_trainer.py",
    }
    if (
        initialization.get("schema") != PALETTE_INIT_SCHEMA
        or initialization.get("status") != "COMPLETE_NONDEGENERATE_FIT"
        or initialization.get("score_claim") is not False
        or initialization.get("inherited_palette_provenance")
        != "VIDEO-DERIVED, CE-TRAINED, INHERITED"
    ):
        raise QBT1Error("QBT2B launch request received an invalid initialization receipt")
    fit = json.loads(Path(initialization["fit_receipt"]["path"]).read_text(encoding="utf-8"))
    if fit.get("degenerate") is not False:
        raise QBT1Error("QBT2B readout fit is degenerate")
    if (
        smoke.get("schema") != RESULT_SCHEMA
        or smoke.get("status") != "BIRTH_STAGE_CAPPED_WITHOUT_EVENT_HANDOFF"
        or smoke.get("score_claim") is not False
        or smoke.get("training_launch") is not False
        or smoke.get("resume_identity_across_03a_03", {}).get("bit_faithful") is not True
        or smoke.get("all_payloads_retained") is not True
    ):
        raise QBT1Error("QBT2B launch request received a stale or invalid bounded smoke")
    if (
        review.get("schema") != "ddm_qbt1_two_pass_review_receipt.v1"
        or set(review.get("python_files", ())) != expected_review_files
        or review.get("passes_per_file") != 2
        or review.get("status") != "PASS"
    ):
        raise QBT1Error("QBT2B launch request lacks the required two-pass Python review receipt")
    initialized_state = Path(initialization["initialized_state"]["path"])
    config = compile_qbt2b_config(
        action="train",
        output=TRAIN_ROOT / "governed_n32_r3",
        pair_ids=SELECTION_IDS,
        device="mps",
        initialization_state=initialized_state,
        birth_max_steps=QBT2B_BIRTH_MAX_STEPS,
        margin_steps=QBT2B_MARGIN_STEPS,
    )
    config["launch_authorized"] = False
    config["scorer_lane"] = {"claimed": False, "claim_id": None}
    config["metal_lane"] = {"claimed": False, "claim_id": None}
    config_path = output_path.parent / "COMPILED_N32_R3_CONFIG.json"
    atomic_json(config_path, config)
    storage = project_on_disk_storage(
        smoke,
        output=output_path.parent,
        total_steps=QBT2B_TOTAL_STEPS,
        checkpoint_every_steps=int(config["checkpoint_every_steps"]),
        birth_max_steps=QBT2B_BIRTH_MAX_STEPS,
        birth_verdict_every_steps=int(config["birth_verdict_every_steps"]),
    )
    memory = smoke["memory_projection"]
    smoke_steps = max(1, len(smoke["history"]))
    smoke_pairs = max(1, max(int(row["materialized_pairs"]) for row in smoke["history"]))
    projected_wall = (
        float(smoke["elapsed_seconds"])
        / smoke_steps
        * QBT2B_TOTAL_STEPS
        * REAL_TRAIN_CHUNK_PAIRS
        / smoke_pairs
    )
    blockers = []
    if not memory["passes_ceiling"]:
        blockers.append("projected materialization peak exceeds 116 GiB")
    if not storage["passes_live_df"]:
        blockers.append("cluster-aware retained-output demand exceeds live AP free space")
    blockers.append(
        "stage-05 same-budget QBW1 control remains absent; stages 03a/03/04 may fire, but stage 05 cannot admit without the real retained control"
    )
    request = {
        "schema": "ddm_qbt2b_sealed_r3_fire_order.v1",
        "disposition": (
            "QUEUED_R3_STAGE03A_03_04_FIRE_STAGE05_BLOCKED"
            if len(blockers) == 1
            else "BLOCKED_NOT_LAUNCHABLE"
        ),
        "owner": "MAIN QBFLOW r3 inherited-palette CE-birth owner",
        "consumer_store": str((TRAIN_ROOT / "governed_n32_r3").resolve()),
        "fire_trigger": (
            "MAIN verifies the committed serializer hashes and two-pass review receipt, re-reads live AP df, confirms no duplicate full-scorer or Metal lane, claims both lanes, then writes launch_authorized=true plus the real claim IDs into a copied config and fires stage 03a"
        ),
        "stage_03_handoff_trigger": (
            "all five classes are present with within-class error <0.20 for two consecutive realized n32 verdicts; otherwise stop at the 100-step safety cap without entering margin training"
        ),
        "stage_05_fire_trigger": (
            "the governed n32 r3 result exists and a real retained same-budget QBW1 control passes custody, pair-set, budget, and score-arithmetic validation"
        ),
        "compiled_config": file_fact(config_path),
        "initialization_receipt": file_fact(initialization_result_path),
        "palette_custody": initialization["palette_custody"],
        "readout_fit_receipt": initialization["fit_receipt"],
        "init_gate": initialization["init_gate"],
        "bounded_smoke": file_fact(smoke_result_path),
        "two_pass_review_receipt": file_fact(review_receipt_path),
        "memory_projection": memory,
        "on_disk_storage_projection": storage,
        "schedule_estimate": {
            "birth_safety_cap_updates": QBT2B_BIRTH_MAX_STEPS,
            "margin_updates_after_event": QBT2B_MARGIN_STEPS,
            "maximum_optimizer_updates": QBT2B_TOTAL_STEPS,
            "wall_seconds_upper_projection": projected_wall,
            "basis": "n1 CPU birth-smoke elapsed scaled linearly to 16-pair chunks and the maximum 5,100-update schedule; not a Metal measurement",
            "status": "DERIVED_CONSERVATIVE_PROJECTION_REMEASURE_ON_METAL_BEFORE_FIRE",
        },
        "fire_order_checklist": {
            "frozen_qbf1_abi_verified": True,
            "inherited_palette_sha_pinned_and_labeled": True,
            "data_dependent_readout_fit_nondegenerate": True,
            "init_n32_realized_receipt_retained": True,
            "init_gate_passed": bool(initialization["init_gate"]["passed"]),
            "init_gate_miss_is_nonblocking": True,
            "realized_ce_birth_before_margin": True,
            "pose_active_during_birth": True,
            "realized_event_handoff_stable_twice": True,
            "chunk_pairs_le_30_structural": True,
            "ema_lawref_resolved_for_5100_max_updates": True,
            "resume_identity_across_03a_03": True,
            "checkpoint_reencode_one_tar_default": True,
            "cluster_aware_storage_projection_passed": bool(storage["passes_live_df"]),
            "memory_projection_under_116_gib": bool(memory["passes_ceiling"]),
            "same_budget_qbw1_control_bound": False,
            "metal_lane_claimed": False,
            "scorer_lane_claimed": False,
            "arm_two_pass_review_complete": True,
        },
        "blockers": blockers,
        "training_launched": False,
        "metal_invocations": 0,
        "modal_invocations": 0,
        "contest_eval_invocations": 0,
        "score_claim": False,
    }
    atomic_json(output_path, request)
    return request


def compile_r7_authorized_config(
    path: Path,
    *,
    smoke_result_path: Path = R7_RETENTION_ROOT / "smoke_n1/RESULT.json",
) -> dict[str, Any]:
    """Seal the r7 treatment config; MAIN must add live claims before firing it."""

    config = compile_qbt2b_config(
        action="train",
        output=TRAIN_ROOT / "governed_n32_r7",
        pair_ids=SELECTION_IDS,
        device="mps",
        initialization_state=(
            TRAIN_ROOT / "governed_n32_r5/initialized_r6_from_r5_cap_ema_state.pt"
        ),
        birth_max_steps=20,
        margin_steps=QBT2B_MARGIN_STEPS,
        birth_class_weight_mode="balanced",
        birth_event_mode="existence_majority",
        margin_constraint_mode=MARGIN_CONSTRAINT_LANE_MOVABLE,
    )
    # The arm owns the sealed treatment tuple, not dispatch authority.  These
    # fields are excluded from config_identity, so MAIN can bind real claims
    # without changing the single treatment variable or cross-mode resume law.
    config["launch_authorized"] = False
    config["scorer_lane"] = {"claimed": False, "claim_id": None}
    config["metal_lane"] = {"claimed": False, "claim_id": None}
    validate_config(config, require_launch_authority=False)
    config_fact = atomic_json(path, config)
    roundtrip = json.loads(path.read_text(encoding="utf-8"))
    validate_config(roundtrip, require_launch_authority=False)
    if roundtrip != config or canonical_sha256(roundtrip) != canonical_sha256(config):
        raise QBT1Error("r7 authorized config JSON round-trip differs")
    smoke_result = json.loads(smoke_result_path.read_text(encoding="utf-8"))
    storage_projection = project_on_disk_storage(
        smoke_result,
        output=TRAIN_ROOT / "governed_n32_r7",
        total_steps=20 + QBT2B_MARGIN_STEPS,
        checkpoint_every_steps=int(config["checkpoint_every_steps"]),
        birth_max_steps=20,
        birth_verdict_every_steps=int(config["birth_verdict_every_steps"]),
    )
    storage_projection_fact = atomic_json(
        R7_RETENTION_ROOT / "R7_STORAGE_PROJECTION_20260828.json", storage_projection
    )
    return {
        "schema": "ddm_qbt2b_r7_authorized_config_compile.v1",
        "status": (
            "SEALED_AWAITING_MAIN_LIVE_CLAIMS"
            if storage_projection["passes_live_df"]
            else "SEALED_BLOCKED_LIVE_STORAGE_PREFLIGHT"
        ),
        "config": config_fact,
        "config_identity_sha256": canonical_sha256(config_identity(config)),
        "config_canonical_sha256": canonical_sha256(config),
        "json_roundtrip_identical": True,
        "validated_before_write": True,
        "validated_after_read": True,
        "storage_projection": storage_projection_fact,
        "storage_projection_passed": bool(storage_projection["passes_live_df"]),
        "launch_authorized": False,
        "score_claim": False,
    }


# r8 continuation: extend the PROVEN constrained margin law from the r7 stage-03
# endpoint.  margin_steps=15_000 is DERIVED from the r7 trajectory fit (flip
# follows a power law ~9.94*t^-0.781 over the second-half window means with a
# zero-floor exponential fit; verdict memo
# ddm_qbt2b_r7_constrained_margin_verdict_20260828.md): the extension buys
# three more power-law doublings toward the half-r6 milestone flip~0.005 at
# cumulative 20k steps and tests exponent stability before any n600 decision.
R8_MARGIN_STEPS = 15_000
R7_STAGE3_END_CHECKPOINT = (
    TRAIN_ROOT
    / "governed_n32_r7/stage_03_joint_boundary_interior_birth/checkpoints/stage_03_end.pt"
)
R8_INITIALIZATION_STATE = (
    TRAIN_ROOT / "governed_n32_r7/initialized_r8_from_r7_stage03_end_ema_state.pt"
)
R8_RETENTION_ROOT = Path(
    "/Volumes/APDataStore/pact/ddm_qbt2b_r8_constrained_margin_continuation"
)


def build_r8_initialized_state(
    source_checkpoint: Path = R7_STAGE3_END_CHECKPOINT,
    output_path: Path = R8_INITIALIZATION_STATE,
) -> dict[str, Any]:
    """Extract the r7 stage-03 endpoint EMA shadow as the r8 initialization state.

    The lineage always initializes from the EMA shadow, never live weights.
    The emitted schema is the exact loader contract, so the strict=True load at
    model construction is the round-trip proof.
    """

    checkpoint = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("stage") != "stage_03_joint_boundary_interior_birth_end":
        raise QBT1Error("r8 init source is not the r7 stage-03 end checkpoint")
    shadow = checkpoint["ema"]["shadow"]
    reference = load_initial_model(torch.device("cpu"))
    if set(shadow) != set(reference.state_dict()):
        raise QBT1Error("r7 EMA shadow keys differ from the QBF1 model state")
    state = {
        "schema": "ddm_qbt2b_initialized_qbf1_state.v1",
        "state_dict": {
            name: value.detach().cpu().clone() for name, value in shadow.items()
        },
        "provenance": {
            "source_checkpoint": file_fact(source_checkpoint),
            "source_stage": str(checkpoint.get("stage")),
            "source_step": int(checkpoint.get("step", -1)),
            "basis": "ema_shadow",
            "shadowed_tensors": sorted(shadow),
        },
    }
    return atomic_torch(output_path, state)


def compile_r8_authorized_config(
    path: Path,
    *,
    smoke_result_path: Path = R7_RETENTION_ROOT / "smoke_n1/RESULT.json",
) -> dict[str, Any]:
    """Seal the r8 continuation config; MAIN must add live claims before firing it.

    r8 differs from r7 in exactly two SCOPE fields: the initialization state
    (the r7 stage-03 EMA endpoint) and margin_steps (5,000 -> 15,000, derived
    from the r7 trajectory fit).  Every mechanism pin — the constraint tuple,
    birth law, and curriculum — is identical to r7's.  One RETENTION-only field
    also moves: checkpoint_every_steps rides the crash-loss cadence law
    (steps // CHECKPOINT_CRASH_LOSS_DENOMINATOR = 50) because r7's every-5
    cadence projects ~146 GB of retained checkpoints at 15,020 steps against
    ~50 GB free — retention cadence gates saves only, never training dynamics.
    """

    return compile_continuation_config(
        path,
        initialization_state=R8_INITIALIZATION_STATE,
        margin_steps=R8_MARGIN_STEPS,
        run_name="governed_n32_r8",
        storage_projection_path=R8_RETENTION_ROOT / "R8_STORAGE_PROJECTION_20260828.json",
        smoke_result_path=smoke_result_path,
        receipt_schema="ddm_qbt2b_r8_authorized_config_compile.v1",
    )


def compile_continuation_config(
    path: Path,
    *,
    initialization_state: Path,
    margin_steps: int,
    run_name: str,
    storage_projection_path: Path,
    smoke_result_path: Path = R7_RETENTION_ROOT / "smoke_n1/RESULT.json",
    receipt_schema: str = "ddm_qbt2b_continuation_authorized_config_compile.v1",
    checkpoint_history_mode: str = "embedded",
) -> dict[str, Any]:
    """Seal a warm-start continuation config (the generalized r7->r8->r9... round).

    Every continuation round differs from its predecessor in exactly two SCOPE
    fields — the initialization state (the predecessor's stage-03 EMA endpoint)
    and margin_steps — plus the RETENTION-only cadence, which is derived here
    from the crash-loss law (steps // CHECKPOINT_CRASH_LOSS_DENOMINATOR).  Every
    mechanism pin (constraint tuple, birth law, curriculum) is held fixed by
    compile_qbt2b_config; the storage projection gates the seal fail-closed.
    """

    if not initialization_state.exists():
        raise QBT1Error("continuation initialization state is absent; build the init first")
    total_steps = 20 + int(margin_steps)
    config = compile_qbt2b_config(
        action="train",
        output=TRAIN_ROOT / run_name,
        pair_ids=SELECTION_IDS,
        device="mps",
        initialization_state=initialization_state,
        birth_max_steps=20,
        margin_steps=int(margin_steps),
        birth_class_weight_mode="balanced",
        birth_event_mode="existence_majority",
        margin_constraint_mode=MARGIN_CONSTRAINT_LANE_MOVABLE,
        checkpoint_every_steps=max(1, total_steps // CHECKPOINT_CRASH_LOSS_DENOMINATOR),
        checkpoint_history_mode=checkpoint_history_mode,
    )
    config["launch_authorized"] = False
    config["scorer_lane"] = {"claimed": False, "claim_id": None}
    config["metal_lane"] = {"claimed": False, "claim_id": None}
    validate_config(config, require_launch_authority=False)
    config_fact = atomic_json(path, config)
    roundtrip = json.loads(path.read_text(encoding="utf-8"))
    validate_config(roundtrip, require_launch_authority=False)
    if roundtrip != config or canonical_sha256(roundtrip) != canonical_sha256(config):
        raise QBT1Error("continuation authorized config JSON round-trip differs")
    smoke_result = json.loads(smoke_result_path.read_text(encoding="utf-8"))
    storage_projection = project_on_disk_storage(
        smoke_result,
        output=TRAIN_ROOT / run_name,
        total_steps=total_steps,
        checkpoint_every_steps=int(config["checkpoint_every_steps"]),
        birth_max_steps=20,
        birth_verdict_every_steps=int(config["birth_verdict_every_steps"]),
        history_mode=str(config.get("checkpoint_history_mode", "embedded")),
    )
    storage_projection_fact = atomic_json(storage_projection_path, storage_projection)
    return {
        "schema": receipt_schema,
        "status": (
            "SEALED_AWAITING_MAIN_LIVE_CLAIMS"
            if storage_projection["passes_live_df"]
            else "SEALED_BLOCKED_LIVE_STORAGE_PREFLIGHT"
        ),
        "config": config_fact,
        "config_identity_sha256": canonical_sha256(config_identity(config)),
        "config_canonical_sha256": canonical_sha256(config),
        "json_roundtrip_identical": True,
        "validated_before_write": True,
        "validated_after_read": True,
        "storage_projection": storage_projection_fact,
        "storage_projection_passed": bool(storage_projection["passes_live_df"]),
        "launch_authorized": False,
        "score_claim": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    smoke = sub.add_parser("smoke", help="run the bounded real CPU mechanism smoke")
    smoke.add_argument("--output", type=Path, default=TRAIN_ROOT / "smoke_n1")
    smoke.add_argument("--pairs", type=int, default=1, choices=range(1, SMOKE_MAX_PAIRS + 1))
    smoke.add_argument("--steps", type=int, default=1)
    palette_init = sub.add_parser(
        "prepare-inherited-palette-init",
        help="fit and retain the QBT2B inherited-palette initialization receipt",
    )
    palette_init.add_argument("--output", type=Path, default=QBT2B_ROOT / "initialization")
    birth_smoke = sub.add_parser(
        "birth-smoke", help="run a bounded n<=4 CE-birth stage and 03a/03 resume-identity smoke"
    )
    birth_smoke.add_argument("--output", type=Path, default=QBT2B_ROOT / "birth_smoke_n1")
    birth_smoke.add_argument("--pairs", type=int, default=1, choices=range(1, SMOKE_MAX_PAIRS + 1))
    birth_smoke.add_argument(
        "--initialization-state",
        type=Path,
        default=QBT2B_ROOT / "initialization/initialized/initialized_r3_state.pt",
    )
    constraint_smoke = sub.add_parser(
        "constraint-smoke",
        help="run the bounded n1 real constrained-margin law and resume-identity smoke",
    )
    constraint_smoke.add_argument(
        "--output",
        type=Path,
        default=R7_RETENTION_ROOT / "smoke_n1",
    )
    constraint_smoke.add_argument(
        "--pair-id", type=int, default=62, choices=SELECTION_IDS
    )
    constraint_smoke.add_argument(
        "--initialization-state",
        type=Path,
        default=(TRAIN_ROOT / "governed_n32_r5/initialized_r6_from_r5_cap_ema_state.pt"),
    )
    compile_r7 = sub.add_parser(
        "compile-r7-config", help="seal the unlaunched r7 config for MAIN claim binding"
    )
    compile_r7.add_argument(
        "--output",
        type=Path,
        default=TRAIN_ROOT / "AUTHORIZED_N32_R7_5020_20260828.json",
    )
    compile_r7.add_argument(
        "--smoke-result",
        type=Path,
        default=R7_RETENTION_ROOT / "smoke_n1/RESULT.json",
    )
    build_r8 = sub.add_parser(
        "build-r8-init", help="extract the r7 stage-03 EMA endpoint as the r8 init state"
    )
    build_r8.add_argument("--source", type=Path, default=R7_STAGE3_END_CHECKPOINT)
    build_r8.add_argument("--output", type=Path, default=R8_INITIALIZATION_STATE)
    compile_r8 = sub.add_parser(
        "compile-r8-config",
        help="seal the unlaunched r8 continuation config for MAIN claim binding",
    )
    compile_r8.add_argument(
        "--output",
        type=Path,
        default=TRAIN_ROOT / "AUTHORIZED_N32_R8_15020_20260828.json",
    )
    compile_r8.add_argument(
        "--smoke-result",
        type=Path,
        default=R7_RETENTION_ROOT / "smoke_n1/RESULT.json",
    )
    compile_cont = sub.add_parser(
        "compile-continuation-config",
        help="seal an unlaunched warm-start continuation config (generalized r-round)",
    )
    compile_cont.add_argument("--init-state", type=Path, required=True)
    compile_cont.add_argument("--margin-steps", type=int, required=True)
    compile_cont.add_argument("--run-name", required=True)
    compile_cont.add_argument("--output", type=Path, required=True)
    compile_cont.add_argument("--storage-projection", type=Path, required=True)
    compile_cont.add_argument(
        "--smoke-result",
        type=Path,
        default=R7_RETENTION_ROOT / "smoke_n1/RESULT.json",
    )
    compile_cont.add_argument(
        "--history-mode",
        choices=("embedded", "sidecar"),
        default="embedded",
        help="sidecar drops embedded history from periodic checkpoints (O(steps^2) wall cure)",
    )
    compile_request = sub.add_parser("compile-launch-request")
    compile_request.add_argument("--smoke-result", type=Path, required=True)
    compile_request.add_argument("--review-receipt", type=Path, required=True)
    compile_request.add_argument("--output", type=Path, default=TRAIN_ROOT / "COMPILED_LAUNCH_REQUEST.json")
    compile_qbt2b = sub.add_parser("compile-qbt2b-launch-request")
    compile_qbt2b.add_argument("--initialization-result", type=Path, required=True)
    compile_qbt2b.add_argument("--smoke-result", type=Path, required=True)
    compile_qbt2b.add_argument("--review-receipt", type=Path, required=True)
    compile_qbt2b.add_argument(
        "--output", type=Path, default=QBT2B_ROOT / "sealed_r3/SEALED_R3_FIRE_ORDER.json"
    )
    run_config = sub.add_parser("run-config", help="MAIN-only governed execution of an authorized config")
    run_config.add_argument("config", type=Path)
    validate = sub.add_parser("validate-config")
    validate.add_argument("config", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.action == "prepare-inherited-palette-init":
        result = prepare_inherited_palette_initialization(args.output)
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
        return 0
    if args.action == "birth-smoke":
        started = time.monotonic()
        config = compile_qbt2b_config(
            action="smoke",
            output=args.output,
            pair_ids=SELECTION_IDS[: args.pairs],
            device="cpu",
            initialization_state=args.initialization_state,
            birth_max_steps=1,
            margin_steps=1,
        )
        result = run_training(config)
        result["elapsed_seconds"] = time.monotonic() - started
        atomic_json(Path(config["output"]) / "RESULT.json", result)
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
        return 0
    if args.action == "constraint-smoke":
        started = time.monotonic()
        config = compile_qbt2b_config(
            action="smoke",
            output=args.output,
            pair_ids=(args.pair_id,),
            device="cpu",
            initialization_state=args.initialization_state,
            birth_max_steps=10,
            margin_steps=2,
            birth_class_weight_mode="balanced",
            birth_event_mode="existence_majority",
            margin_constraint_mode=MARGIN_CONSTRAINT_LANE_MOVABLE,
        )
        atomic_json(args.output / "SMOKE_CONFIG.json", config)
        result = run_training(config)
        result["elapsed_seconds"] = time.monotonic() - started
        atomic_json(Path(config["output"]) / "RESULT.json", result)
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
        return 0
    if args.action == "compile-r7-config":
        receipt = compile_r7_authorized_config(
            args.output, smoke_result_path=args.smoke_result
        )
        print(json.dumps(receipt, indent=2, sort_keys=True, default=str))
        return 0
    if args.action == "build-r8-init":
        receipt = build_r8_initialized_state(args.source, args.output)
        print(json.dumps(receipt, indent=2, sort_keys=True, default=str))
        return 0
    if args.action == "compile-r8-config":
        receipt = compile_r8_authorized_config(
            args.output, smoke_result_path=args.smoke_result
        )
        print(json.dumps(receipt, indent=2, sort_keys=True, default=str))
        return 0
    if args.action == "compile-continuation-config":
        receipt = compile_continuation_config(
            args.output,
            initialization_state=args.init_state,
            margin_steps=args.margin_steps,
            run_name=args.run_name,
            storage_projection_path=args.storage_projection,
            smoke_result_path=args.smoke_result,
            checkpoint_history_mode=args.history_mode,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True, default=str))
        return 0
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
    if args.action == "compile-qbt2b-launch-request":
        request = compiled_qbt2b_launch_request(
            initialization_result_path=args.initialization_result,
            smoke_result_path=args.smoke_result,
            review_receipt_path=args.review_receipt,
            output_path=args.output,
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
