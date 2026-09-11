# SPDX-License-Identifier: MIT
"""OBX2 edge-local implicit correction burn — stage runner.

This is a local macOS-CPU advisory instrument, not a contest evaluator.  It
executes the governed stages of
`.omx/research/ddm_obx2_edge_local_implicit_correction_burn_spec_20260911.md`
over the complete 600-pair population.

Stage 0 pins every frozen input, proves the retained QBT2B r10 packet/archive
identity, and refuses unless the chartered Vertigo custody root has room for two
projected runs plus reserve.

Stage 2a is a DECLARED addition to the sealed stage list.  It is object-free:
it never builds, trains, or admits a candidate.  It prices the spec's own
`distortion < 0.04` gate by measuring the retained move-44 teacher and a ladder
of deterministic teacher corruptions through the same frozen CPU scorer that
every later stage uses.  Its purpose is the CLAUDE.md Carmack MVP-first rule: a
free local smoke that falsifiably challenges the burn's central assumption
before days of training are spent.  The decisive rung is `sp_384x512`: an
achievable [0,255] render on the QBF grid, back-projected so that its camera
image matches the teacher in the scorer's own plane.  Because it is a
construction, its distortion is an UPPER BOUND on the minimum distortion of any
renderer on that grid, so a comfortable pass clears the grid and a failure
bounds -- but does not by itself close -- the grid.

Every measurement here is `[macOS-CPU advisory]`; `score_claim` is false and no
row is promotable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import resource
import sys
import tarfile
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
for _root in (REPO, SRC):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from experiments import ddm_qbflow_packet as qbf1
from experiments import ddm_qbt1_qbflow_trainer as qbt1
from experiments import ddm_qbz1_descent_rate_configuration as qbz1
from tac.gt_lineage import AUTHORITY_LINEAGE, assert_gt_lineage
from tac.scorer import load_differentiable_scorers

SCHEMA_STAGE0 = "ddm_obx2_stage0_identity.v1"
SCHEMA_STAGE2A = "ddm_obx2_stage2a_gate_pricing.v1"
CHUNK_SCHEMA = "ddm_obx2_stage2a_chunk.v1"

N = 600
EVAL_H, EVAL_W = 384, 512
CAMERA_H, CAMERA_W = 874, 1164
CHANNELS = 3
CHUNK_PAIRS = 20
RATE_DENOMINATOR = 37_545_489
MASTER_SEED = 20260911
SP_ITERATIONS = 12
SP_STEP = 1.0

# Spec gates (burn spec "Exact admission arithmetic").
PACKET_BYTE_GATE = 122_000
DISTORTION_GATE = 0.04
RATE_AT_GATE = 25.0 * PACKET_BYTE_GATE / RATE_DENOMINATOR
DISTORTION_BUDGET_FOR_SUB_012 = 0.12 - RATE_AT_GATE

# Frozen input pins (burn spec "Frozen input pins").
QBT2B_ARCHIVE_BYTES = 106_714
QBT2B_PACKET_BYTES = 106_606
QBT2B_ARCHIVE_SHA256 = "b26371e50696bdcdafdccbf4c629ef1119ae48aa1ac8765200a6ea2176f91830"
QBT2B_PACKET_SHA256 = "607abebda2708f00daab79aac7bc6839d314096e6ed5693b642525487a1019f7"
QBT2B_CONTAINER_SHA256 = "18d69e4da2024d39ef13e73ef92623ca9857e67cc4f7b551f83d557f9880709d"
POINTER_ARCHIVE_BYTES = 180_406
POINTER_ARCHIVE_SHA256 = "04758c0dfb8d94ebe801602aac96d93a4f260aad24f58b6b9cdbbe2270ad460e"
POINTER_RAW_SHA256 = "2b762eba4a20a315c104f8447d6ea0e604f73c3d8b8b69b3fc63b0fc792d59fc"
POINTER_FIELD_SHA256 = "a92e7d902a4498961217f02c2b90d3fb9025901ba6d047201ff3bf297fa2f7a8"
OBX1_RESULT_BYTES = 24_379
OBX1_RESULT_SHA256 = "ae19bb553e6810b4492d2c8df8ba5cec4f319534bf7a559010faa46f9944b81e"

# Pointer authority components (contest-CUDA T4 n600, move 44) used only as the
# cross-axis calibration control.  They are never mixed into an advisory row.
POINTER_AUTHORITY_D_SEG = 0.00010345
POINTER_AUTHORITY_D_POSE = 4.59e-6

OUTPUT_ROOT = Path("/Volumes/VertigoDataTier/pact/ddm_obx2_edge_local_implicit_correction")
QBT2B_CONTAINER = Path(
    "/Volumes/APDataStore/pact/ddm_qbflow_implicit_boundary_flow/qbt1_trainer/"
    "governed_n32_r10/stage_05_same_budget_admission/reencoded/stage_05_end/"
    "reencode_payloads.tar"
)
POINTER_ARCHIVE = Path("/Volumes/VertigoDataTier/pact/ddm_rlc5_cure_on_move43/candidate_runtime/archive.zip")
POINTER_RAW = Path("/Volumes/VertigoDataTier/pact/ddm_rlc5_cure_on_move43/public_rlc4/output/0.raw")
POINTER_FIELD = Path("/Volumes/VertigoDataTier/pact/ddm_sj1_pass6/retained/fields/subset6.u8")
OBX1_RESULT = Path("/Volumes/VertigoDataTier/pact/ddm_obx1_successor_object/RESULT.json")
ACTIVE_CLAIMS = REPO / ".omx/state/active_lane_dispatch_claims.md"
LANE_ID = "ddm_obx2_edge_local_implicit_correction_20260911"

MINIMUM_FREE_BYTES = 20_000_000_000
CAMERA_CHUNK_BYTES = CHUNK_PAIRS * 2 * CAMERA_H * CAMERA_W * CHANNELS
LOGITS_CHUNK_BYTES = CHUNK_PAIRS * 5 * EVAL_H * EVAL_W * 2

# Rungs whose complete corrupted camera payload is retained verbatim.  Every
# other rung is certified-rebuildable: its exact deterministic recipe, master
# seed, and per-chunk SHA-256 of the materialized camera bytes are recorded, so
# any future consumer can reproduce and prove the identical bytes from the
# retained teacher.  No measured payload is reduced to a scalar.
RETAIN_CAMERA_RUNGS = ("sp_384x512", "grid_384x512")
# The sp_384x512 render IS the Stage-2 distillation target, so it is retained.
RETAIN_RENDER_RUNGS = ("sp_384x512",)
RETAIN_LOGITS_RUNGS = ("teacher", "sp_384x512", "grid_384x512", "noise_4")


class OBX2Error(RuntimeError):
    """Fail-closed refusal for OBX2 custody, pin, denominator, or gate drift."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_fact(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise OBX2Error(f"required file is absent: {path}")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest}


def require_fact(path: Path, *, digest: str, size: int | None = None) -> dict[str, Any]:
    fact = file_fact(path)
    if fact["sha256"] != digest or (size is not None and fact["bytes"] != size):
        raise OBX2Error(f"frozen input drifted: {path}")
    return fact


def retain_exact(path: Path, payload: bytes) -> dict[str, Any]:
    if path.exists():
        fact = file_fact(path)
        if fact["bytes"] != len(payload) or fact["sha256"] != sha256_bytes(payload):
            raise OBX2Error(f"existing retained payload differs; refusing overwrite: {path}")
        return fact
    return qbt1.atomic_bytes(path, payload)


def free_bytes(path: Path) -> int:
    stat = os.statvfs(path)
    return int(stat.f_bavail * stat.f_frsize)


def storage_preflight(output: Path, *, required: int) -> dict[str, Any]:
    if output.resolve() != OUTPUT_ROOT.resolve():
        raise OBX2Error(f"output must be the chartered OBX2 custody root: {output.resolve()}")
    output.mkdir(parents=True, exist_ok=True)
    free = free_bytes(output)
    if free < required:
        raise OBX2Error(f"Vertigo storage preflight refused: free={free} required={required}")
    return {
        "root": str(output.resolve()),
        "free_bytes": free,
        "required_free_bytes": required,
        "status": "PASS",
        "cleanup": "certify-or-block; this runner never deletes retained OBX2 evidence",
    }


def assert_active_claim(claim_id: str) -> dict[str, Any]:
    """Refuse unless an OBX2-owned lane row is the newest row for its lane."""

    if not claim_id.startswith("ddm_obx2_"):
        raise OBX2Error("claim id must be an OBX2-owned lane id")
    newest: dict[str, str] | None = None
    for line in ACTIVE_CLAIMS.read_text().splitlines():
        if not line.startswith("| 20"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 7 or cells[2] != claim_id:
            continue
        if newest is None:
            newest = {"timestamp_utc": cells[0], "agent": cells[1], "status": cells[6]}
    if newest is None:
        raise OBX2Error(f"no active lane claim row for {claim_id}")
    if not newest["status"].startswith(("building", "active")):
        raise OBX2Error(f"newest lane row for {claim_id} is terminal: {newest['status']}")
    return newest


def projected_run_bytes(rungs: Sequence[str]) -> dict[str, Any]:
    """Worst-case Stage-2a volume from the instrumented chunk geometry."""

    chunks = math.ceil(N / CHUNK_PAIRS)
    camera = len([name for name in RETAIN_CAMERA_RUNGS if name in rungs]) * chunks * CAMERA_CHUNK_BYTES
    logits = len([name for name in RETAIN_LOGITS_RUNGS if name in rungs]) * chunks * LOGITS_CHUNK_BYTES
    argmax = len(rungs) * chunks * CHUNK_PAIRS * EVAL_H * EVAL_W
    pose = len(rungs) * chunks * CHUNK_PAIRS * 6 * 4
    total = camera + logits + argmax + pose
    return {
        "chunks": chunks,
        "camera_bytes": camera,
        "logits_bytes": logits,
        "argmax_bytes": argmax,
        "pose_bytes": pose,
        "projected_run_bytes": total,
        "two_runs_plus_reserve_bytes": 2 * total + MINIMUM_FREE_BYTES,
    }


def stage0(output: Path, *, rungs: Sequence[str]) -> dict[str, Any]:
    projection = projected_run_bytes(rungs)
    storage = storage_preflight(output, required=projection["two_runs_plus_reserve_bytes"])
    container_fact = require_fact(QBT2B_CONTAINER, digest=QBT2B_CONTAINER_SHA256)
    base_archive = qbt1_tar_member(QBT2B_CONTAINER, "archive.zip")
    base_archive_repeat = qbt1_tar_member(QBT2B_CONTAINER, "archive.repeat.zip")
    base_packet = qbt1_tar_member(QBT2B_CONTAINER, "packet.qbf")
    base_packet_repeat = qbt1_tar_member(QBT2B_CONTAINER, "packet.repeat.qbf")
    if (
        len(base_archive) != QBT2B_ARCHIVE_BYTES
        or sha256_bytes(base_archive) != QBT2B_ARCHIVE_SHA256
        or len(base_packet) != QBT2B_PACKET_BYTES
        or sha256_bytes(base_packet) != QBT2B_PACKET_SHA256
    ):
        raise OBX2Error("QBT2B r10 archive or packet identity differs")
    if base_archive_repeat != base_archive or base_packet_repeat != base_packet:
        raise OBX2Error("QBT2B r10 deterministic repeat differs")
    if qbf1.read_deterministic_archive(base_archive) != base_packet:
        raise OBX2Error("QBT2B r10 archive receiver does not return the retained packet")
    decoded = qbf1.decode_packet(base_packet)
    expected_sections = {
        qbf1.SECTION_CONFIG,
        qbf1.SECTION_MODEL,
        qbf1.SECTION_LATENT_META,
        qbf1.SECTION_LATENTS,
    }
    if set(decoded.sections) != expected_sections:
        raise OBX2Error("receiver-decoded QBF section set differs")
    retained = {
        "base_archive": retain_exact(output / "inputs/base.archive.zip", base_archive),
        "base_archive_repeat": retain_exact(output / "inputs/base.archive.repeat.zip", base_archive_repeat),
        "base_packet": retain_exact(output / "inputs/base.packet.qbf", base_packet),
        "base_packet_repeat": retain_exact(output / "inputs/base.packet.repeat.qbf", base_packet_repeat),
    }
    receipt = {
        "schema": SCHEMA_STAGE0,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "axis": "[macOS-CPU advisory custody/receiver check]",
        "score_claim": False,
        "promotable": False,
        "research_only": True,
        "lane_id": LANE_ID,
        "storage_preflight": storage,
        "projected_volume": projection,
        "source_facts": {
            "qbt2b_container": container_fact,
            "pointer_archive": require_fact(POINTER_ARCHIVE, digest=POINTER_ARCHIVE_SHA256, size=POINTER_ARCHIVE_BYTES),
            "pointer_raw": require_fact(
                POINTER_RAW, digest=POINTER_RAW_SHA256, size=N * 2 * CAMERA_H * CAMERA_W * CHANNELS
            ),
            "pointer_field": require_fact(POINTER_FIELD, digest=POINTER_FIELD_SHA256, size=N * EVAL_H * EVAL_W),
            "obx1_result": require_fact(OBX1_RESULT, digest=OBX1_RESULT_SHA256, size=OBX1_RESULT_BYTES),
        },
        "retained_inputs": retained,
        "packet_section_raw_bytes": {
            qbf1.SECTION_NAMES[section_id]: len(raw) for section_id, raw in sorted(decoded.sections.items())
        },
        "gates": {
            "packet_byte_gate": PACKET_BYTE_GATE,
            "distortion_gate": DISTORTION_GATE,
            "rate_at_packet_byte_gate": RATE_AT_GATE,
            "distortion_budget_for_sub_012": DISTORTION_BUDGET_FOR_SUB_012,
        },
        "host": {"platform": platform.platform(), "python": platform.python_version()},
    }
    qbt1.atomic_json(output / "checkpoints/stage_00_identity.json", receipt)
    return receipt


def qbt1_tar_member(path: Path, member: str) -> bytes:
    with tarfile.open(path, mode="r") as archive:
        stream = archive.extractfile(member)
        if stream is None:
            raise OBX2Error(f"retained tar member is unreadable: {member}")
        return stream.read()


def gate_band(gt_chunk: np.ndarray, radius: int) -> np.ndarray:
    """Four-neighbour argmax-boundary band of the scorer target, dilated `radius`."""

    if gt_chunk.ndim != 3 or gt_chunk.shape[1:] != (EVAL_H, EVAL_W):
        raise OBX2Error("scorer-target geometry differs")
    edge = np.zeros(gt_chunk.shape, dtype=bool)
    edge[:, 1:] |= gt_chunk[:, 1:] != gt_chunk[:, :-1]
    edge[:, :-1] |= gt_chunk[:, 1:] != gt_chunk[:, :-1]
    edge[:, :, 1:] |= gt_chunk[:, :, 1:] != gt_chunk[:, :, :-1]
    edge[:, :, :-1] |= gt_chunk[:, :, 1:] != gt_chunk[:, :, :-1]
    band = edge
    for _ in range(radius):
        grown = band.copy()
        grown[:, 1:] |= band[:, :-1]
        grown[:, :-1] |= band[:, 1:]
        grown[:, :, 1:] |= band[:, :, :-1]
        grown[:, :, :-1] |= band[:, :, 1:]
        band = grown
    return band


def _positive_int(value: str, name: str) -> int:
    if not value.isdigit() or int(value) < 1:
        raise OBX2Error(f"malformed rung amplitude: {name}")
    return int(value)


def rung_specification(name: str) -> dict[str, Any]:
    """Deterministic, self-describing recipe for one Stage-2a ladder rung."""

    if name == "teacher":
        return {"name": name, "kind": "identity", "family": "control"}
    if name.startswith("grid_"):
        parts = name.removeprefix("grid_").split("x")
        if len(parts) != 2 or not all(part.isdigit() and int(part) > 0 for part in parts):
            raise OBX2Error(f"malformed render-grid rung: {name}")
        height, width = (int(part) for part in parts)
        return {
            "name": name,
            "kind": "render_grid",
            "family": "capacity",
            "height": height,
            "width": width,
            "down": "area",
            "up": "bicubic",
        }
    if name.startswith("sp384_render_noise_"):
        return {
            "name": name,
            "kind": "scorer_plane_render_noise",
            "family": "render_capacity",
            "height": EVAL_H,
            "width": EVAL_W,
            "iterations": SP_ITERATIONS,
            "step": SP_STEP,
            "amplitude_lsb": _positive_int(name.removeprefix("sp384_render_noise_"), name),
            "note": (
                "the sp_384x512 render perturbed by seeded uniform integer noise BEFORE the camera "
                "upsample; prices how accurate a generator's own 384x512 output must be"
            ),
        }
    if name.startswith("sp_"):
        parts = name.removeprefix("sp_").split("x")
        if len(parts) != 2 or not all(part.isdigit() and int(part) > 0 for part in parts):
            raise OBX2Error(f"malformed scorer-plane rung: {name}")
        height, width = (int(part) for part in parts)
        return {
            "name": name,
            "kind": "scorer_plane_grid",
            "family": "capacity",
            "height": height,
            "width": width,
            "iterations": SP_ITERATIONS,
            "step": SP_STEP,
            "retain_render": name in RETAIN_RENDER_RUNGS,
            "note": (
                "iterative back-projection of a [0,255]-clamped render on this grid so that its "
                "bicubic camera image matches the teacher IN THE SCORER PLANE; an achievable "
                "construction, hence an upper bound on the minimum distortion of a renderer on "
                "this grid"
            ),
        }
    if name.startswith("noise_"):
        return {
            "name": name,
            "kind": "uniform_noise",
            "family": "photometric",
            "amplitude_lsb": _positive_int(name.removeprefix("noise_"), name),
            "support": "all_camera_pixels",
        }
    if name.startswith("interior_noise_"):
        return {
            "name": name,
            "kind": "uniform_noise",
            "family": "oracle_structured",
            "amplitude_lsb": _positive_int(name.removeprefix("interior_noise_"), name),
            "support": "camera pixels outside the GT argmax boundary band (oracle; not a receiver mechanism)",
            "band_radius_eval_cells": 2,
        }
    raise OBX2Error(f"unknown Stage-2a rung: {name}")


def apply_rung(
    teacher: torch.Tensor,
    spec: Mapping[str, Any],
    *,
    pair_ids: Sequence[int],
    gt_chunk: np.ndarray,
) -> tuple[torch.Tensor, dict[str, Any], dict[str, np.ndarray]]:
    """Materialize one rung's camera tensor [B,2,3,874,1164], diagnostics, and payloads."""

    if teacher.ndim != 5 or teacher.shape[1:] != (2, CHANNELS, CAMERA_H, CAMERA_W):
        raise OBX2Error("teacher camera geometry differs")
    kind = spec["kind"]
    if kind == "identity":
        return teacher.clone(), {}, {}
    batch = teacher.shape[0]
    flat = teacher.reshape(batch * 2, CHANNELS, CAMERA_H, CAMERA_W)
    if kind == "render_grid":
        low = F.interpolate(flat, size=(int(spec["height"]), int(spec["width"])), mode="area")
        high = F.interpolate(low, size=(CAMERA_H, CAMERA_W), mode="bicubic", align_corners=False)
        return high.round().clamp(0.0, 255.0).reshape(batch, 2, CHANNELS, CAMERA_H, CAMERA_W), {}, {}
    if kind == "scorer_plane_grid":
        height, width = int(spec["height"]), int(spec["width"])
        high, history, render = scorer_plane_render(
            flat,
            height=height,
            width=width,
            iterations=int(spec["iterations"]),
            step=float(spec["step"]),
        )
        arrays = {}
        if bool(spec.get("retain_render", False)):
            arrays["render_u8"] = (
                render.reshape(batch, 2, CHANNELS, height, width).to(torch.uint8).cpu().numpy()
            )
        return (
            high.reshape(batch, 2, CHANNELS, CAMERA_H, CAMERA_W),
            {"scorer_plane_residual_rmse_history": history},
            arrays,
        )
    if kind == "scorer_plane_render_noise":
        height, width = int(spec["height"]), int(spec["width"])
        amplitude = int(spec["amplitude_lsb"])
        if amplitude < 1:
            raise OBX2Error("render-noise amplitude must be at least one LSB")
        _, history, render = scorer_plane_render(
            flat,
            height=height,
            width=width,
            iterations=int(spec["iterations"]),
            step=float(spec["step"]),
        )
        noisy = torch.empty_like(render)
        for index, pair_id in enumerate(pair_ids):
            generator = torch.Generator(device="cpu")
            generator.manual_seed(MASTER_SEED * 7_919 + int(pair_id) * 1_009 + amplitude)
            draw = torch.randint(
                -amplitude,
                amplitude + 1,
                (2, CHANNELS, height, width),
                generator=generator,
                dtype=torch.int16,
            ).float()
            noisy[2 * index : 2 * index + 2] = (
                render[2 * index : 2 * index + 2] + draw
            ).round().clamp(0.0, 255.0)
        high = F.interpolate(noisy, size=(CAMERA_H, CAMERA_W), mode="bicubic", align_corners=False)
        high = high.round().clamp(0.0, 255.0)
        return (
            high.reshape(batch, 2, CHANNELS, CAMERA_H, CAMERA_W),
            {"scorer_plane_residual_rmse_history": history},
            {},
        )
    if kind == "uniform_noise":
        amplitude = int(spec["amplitude_lsb"])
        if amplitude < 1:
            raise OBX2Error("noise amplitude must be at least one LSB")
        keep_flat = None
        if spec["family"] == "oracle_structured":
            band = gate_band(gt_chunk, int(spec["band_radius_eval_cells"]))
            keep = torch.from_numpy((~band).astype(np.float32))[:, None]
            keep_camera = F.interpolate(keep, size=(CAMERA_H, CAMERA_W), mode="nearest")
            keep_flat = keep_camera.repeat_interleave(2, dim=0)
        out = torch.empty_like(flat)
        for index, pair_id in enumerate(pair_ids):
            generator = torch.Generator(device="cpu")
            generator.manual_seed(MASTER_SEED * 1_000_003 + int(pair_id) * 1_009 + amplitude)
            draw = torch.randint(
                -amplitude,
                amplitude + 1,
                (2, CHANNELS, CAMERA_H, CAMERA_W),
                generator=generator,
                dtype=torch.int16,
            ).float()
            if keep_flat is not None:
                draw = draw * keep_flat[2 * index : 2 * index + 2]
            out[2 * index : 2 * index + 2] = (
                flat[2 * index : 2 * index + 2] + draw
            ).round().clamp(0.0, 255.0)
        return out.reshape(batch, 2, CHANNELS, CAMERA_H, CAMERA_W), {}, {}
    raise OBX2Error(f"unknown rung kind: {kind}")


def to_scorer_plane(camera_flat: torch.Tensor) -> torch.Tensor:
    """The frozen scorer's own first operation: bilinear 874x1164 -> 384x512."""

    return F.interpolate(camera_flat, size=(EVAL_H, EVAL_W), mode="bilinear", align_corners=False)


def scorer_plane_render(
    teacher_flat: torch.Tensor,
    *,
    height: int,
    width: int,
    iterations: int,
    step: float,
) -> tuple[torch.Tensor, list[float], torch.Tensor]:
    """Best achievable camera image from a [0,255] render on a `height x width` grid.

    Iterative back-projection against the scorer plane.  Returns the rounded uint8
    camera tensor, the measured scorer-plane residual RMSE per iteration (so
    convergence is measured and never assumed), and the uint8-rounded render on
    the requested grid.  The render is rounded before the camera upsample because
    a shipped generator emits integer-valued frames, not real numbers.
    """

    target = to_scorer_plane(teacher_flat)
    render = F.interpolate(teacher_flat, size=(height, width), mode="area").clamp(0.0, 255.0)
    history: list[float] = []
    for _ in range(iterations):
        camera = F.interpolate(render, size=(CAMERA_H, CAMERA_W), mode="bicubic", align_corners=False)
        residual = target - to_scorer_plane(camera.clamp(0.0, 255.0))
        history.append(float(residual.square().mean().sqrt()))
        if height == EVAL_H and width == EVAL_W:
            correction = residual
        else:
            correction = F.interpolate(residual, size=(height, width), mode="bilinear", align_corners=False)
        render = (render + step * correction).clamp(0.0, 255.0)
    render = render.round().clamp(0.0, 255.0)
    camera = F.interpolate(render, size=(CAMERA_H, CAMERA_W), mode="bicubic", align_corners=False)
    camera = camera.round().clamp(0.0, 255.0)
    history.append(float((target - to_scorer_plane(camera)).square().mean().sqrt()))
    return camera, history, render


def score_arrays(
    camera: torch.Tensor,
    *,
    posenet: torch.nn.Module,
    segnet: torch.nn.Module,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pose6, logits = qbt1.scorer_forward(camera, posenet, segnet)
    return (
        pose6.cpu().numpy().astype("<f4"),
        logits.cpu().numpy().astype("<f2"),
        logits.argmax(dim=1).cpu().numpy().astype(np.uint8),
    )


def camera_squared_error(camera_u8: np.ndarray, teacher_np: np.ndarray, index: int) -> int:
    """Exact integer squared error of one rung pair against the teacher pair."""

    got = camera_u8[index].astype(np.int32)
    want = np.transpose(teacher_np[index], (0, 3, 1, 2)).astype(np.int32)
    if got.shape != want.shape:
        raise OBX2Error("rung camera layout differs from the teacher layout")
    difference = got - want
    return int(np.einsum("ijkl,ijkl->", difference, difference, dtype=np.int64))


def chunk_rows(
    *,
    pair_ids: Sequence[int],
    argmax: np.ndarray,
    pose: np.ndarray,
    gt_chunk: np.ndarray,
    pose_target: np.ndarray,
    camera_u8: np.ndarray,
    teacher_np: np.ndarray,
    scorer_plane_squared: Sequence[float],
) -> list[dict[str, Any]]:
    values_per_pair = 2 * CAMERA_H * CAMERA_W * CHANNELS
    scorer_values_per_pair = 2 * EVAL_H * EVAL_W * CHANNELS
    if len(scorer_plane_squared) != len(pair_ids):
        raise OBX2Error("scorer-plane residual count differs from the pair count")
    rows = []
    for index, pair_id in enumerate(pair_ids):
        rows.append(
            {
                "pair_id": int(pair_id),
                "scorer_plane_squared_error_sum": float(scorer_plane_squared[index]),
                "scorer_plane_values": scorer_values_per_pair,
                "seg_errors": int((argmax[index] != gt_chunk[index]).sum()),
                "seg_pixels": int(gt_chunk[index].size),
                "pose_squared_error_sum": float(
                    np.square(pose[index].astype(np.float64) - pose_target[index].astype(np.float64)).sum()
                ),
                "pose_values": 6,
                "camera_squared_error_sum": camera_squared_error(camera_u8, teacher_np, index),
                "camera_values": values_per_pair,
            }
        )
    return rows


def realize_rung_chunk(
    output: Path,
    *,
    rung: str,
    pair_ids: Sequence[int],
    teacher_raw: np.memmap,
    gt: np.ndarray,
    pose_target: np.ndarray,
    posenet: torch.nn.Module,
    segnet: torch.nn.Module,
) -> dict[str, Any]:
    spec = rung_specification(rung)
    stem = f"pairs_{pair_ids[0]:04d}_{pair_ids[-1]:04d}"
    payload_path = output / "stage_2a" / rung / f"{stem}.npz"
    checkpoint_path = output / "checkpoints" / f"stage_02a_{rung}_{stem}.json"
    if checkpoint_path.exists() and payload_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text())
        if checkpoint.get("payload", {}).get("sha256") != file_fact(payload_path)["sha256"]:
            raise OBX2Error(f"resumed Stage-2a payload differs from its checkpoint: {payload_path}")
        checkpoint["resumed"] = True
        return checkpoint

    gt_chunk = np.asarray(gt[list(pair_ids)], dtype=np.uint8)
    target_pose = np.asarray(pose_target[list(pair_ids)], dtype="<f4")
    teacher_np = np.asarray(teacher_raw[list(pair_ids)], dtype=np.uint8)
    teacher = torch.from_numpy(teacher_np.copy()).permute(0, 1, 4, 2, 3).float()
    started = time.time()
    with torch.no_grad():
        camera, diagnostics, extra_arrays = apply_rung(teacher, spec, pair_ids=pair_ids, gt_chunk=gt_chunk)
        camera_u8 = camera.to(torch.uint8).cpu().numpy()
        pairs = camera.shape[0]
        scorer_plane_squared = (
            (
                to_scorer_plane(camera.reshape(pairs * 2, CHANNELS, CAMERA_H, CAMERA_W))
                - to_scorer_plane(teacher.reshape(pairs * 2, CHANNELS, CAMERA_H, CAMERA_W))
            )
            .square()
            .reshape(pairs, -1)
            .sum(dim=1)
            .double()
            .tolist()
        )
        pose, logits, argmax = score_arrays(camera, posenet=posenet, segnet=segnet)
    elapsed = time.time() - started

    arrays: dict[str, np.ndarray] = {
        "pair_ids_i64": np.asarray(pair_ids, dtype=np.int64),
        "segnet_argmax_u8": argmax,
        "posenet_pose6_f32": pose,
        "target_argmax_u8": gt_chunk,
        "target_pose6_f32": target_pose,
    }
    if rung in RETAIN_LOGITS_RUNGS:
        arrays["segnet_logits_f16"] = logits
    if rung in RETAIN_CAMERA_RUNGS:
        arrays["camera_pair_u8"] = camera_u8
    arrays.update(extra_arrays)
    payload_fact = qbt1.atomic_npz(payload_path, **arrays)
    rows = chunk_rows(
        pair_ids=pair_ids,
        argmax=argmax,
        pose=pose,
        gt_chunk=gt_chunk,
        pose_target=target_pose,
        camera_u8=camera_u8,
        teacher_np=teacher_np,
        scorer_plane_squared=scorer_plane_squared,
    )
    checkpoint = {
        "schema": CHUNK_SCHEMA,
        "resumed": False,
        "rung": rung,
        "rung_specification": spec,
        "diagnostics": diagnostics,
        "master_seed": MASTER_SEED,
        "payload": payload_fact,
        "camera_bytes_sha256": sha256_bytes(camera_u8.tobytes(order="C")),
        "camera_retained_verbatim": rung in RETAIN_CAMERA_RUNGS,
        "logits_retained": rung in RETAIN_LOGITS_RUNGS,
        "elapsed_seconds": elapsed,
        "max_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "pair_rows": rows,
    }
    qbt1.atomic_json(checkpoint_path, checkpoint)
    return checkpoint


def aggregate_rung(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ordered = [int(row["pair_id"]) for row in rows]
    if len(rows) != N or ordered != list(range(N)):
        raise OBX2Error("Stage-2a pair denominator is not exactly 600 ordered rows")
    seg_errors = sum(int(row["seg_errors"]) for row in rows)
    seg_pixels = sum(int(row["seg_pixels"]) for row in rows)
    pose_sum = sum(float(row["pose_squared_error_sum"]) for row in rows)
    pose_values = sum(int(row["pose_values"]) for row in rows)
    camera_sum = sum(int(row["camera_squared_error_sum"]) for row in rows)
    camera_values = sum(int(row["camera_values"]) for row in rows)
    scorer_plane_sum = sum(float(row["scorer_plane_squared_error_sum"]) for row in rows)
    scorer_plane_values = sum(int(row["scorer_plane_values"]) for row in rows)
    if scorer_plane_values != N * 2 * EVAL_H * EVAL_W * CHANNELS:
        raise OBX2Error("Stage-2a scorer-plane denominator differs")
    if seg_pixels != N * EVAL_H * EVAL_W or pose_values != N * 6:
        raise OBX2Error("Stage-2a scorer denominator differs")
    if camera_values != N * 2 * CAMERA_H * CAMERA_W * CHANNELS:
        raise OBX2Error("Stage-2a camera denominator differs")
    d_seg = seg_errors / seg_pixels
    d_pose = pose_sum / pose_values
    distortion = 100.0 * d_seg + math.sqrt(10.0 * d_pose)
    return {
        "seg_errors": seg_errors,
        "seg_pixels": seg_pixels,
        "d_seg": d_seg,
        "pose_squared_error_sum": pose_sum,
        "pose_values": pose_values,
        "d_pose": d_pose,
        "distortion": distortion,
        "camera_rmse_vs_teacher": math.sqrt(camera_sum / camera_values),
        "camera_values": camera_values,
        "scorer_plane_rmse_vs_teacher": math.sqrt(scorer_plane_sum / scorer_plane_values),
        "scorer_plane_values": scorer_plane_values,
        "passes_distortion_gate": distortion < DISTORTION_GATE,
        "strict_byte_cap_at_own_distortion": math.floor(
            max(0.0, 0.12 - distortion) * RATE_DENOMINATOR / 25.0
        ),
    }


def stage2a(output: Path, *, rungs: Sequence[str], claim_id: str) -> dict[str, Any]:
    claim = assert_active_claim(claim_id)
    assert_gt_lineage(qbz1.GT_ARGMAX, required=AUTHORITY_LINEAGE, instrument="OBX2 DALI partition")
    assert_gt_lineage(qbz1.GT_POSE6, required=AUTHORITY_LINEAGE, instrument="OBX2 DALI pose")
    gt = np.load(qbz1.GT_ARGMAX, mmap_mode="r", allow_pickle=False)
    pose_target = np.load(qbz1.GT_POSE6, mmap_mode="r", allow_pickle=False)
    if gt.shape != (N, EVAL_H, EVAL_W) or gt.dtype != np.uint8 or pose_target.shape != (N, 6):
        raise OBX2Error("registered scorer-target geometry differs")
    teacher_raw = np.memmap(POINTER_RAW, dtype=np.uint8, mode="r", shape=(N, 2, CAMERA_H, CAMERA_W, CHANNELS))
    torch.manual_seed(MASTER_SEED)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    posenet, segnet = load_differentiable_scorers(REPO / "upstream", device=torch.device("cpu"))
    posenet.eval()
    segnet.eval()

    results: dict[str, Any] = {}
    for rung in rungs:
        rows: list[dict[str, Any]] = []
        chunk_facts = []
        for start in range(0, N, CHUNK_PAIRS):
            pair_ids = list(range(start, min(N, start + CHUNK_PAIRS)))
            checkpoint = realize_rung_chunk(
                output,
                rung=rung,
                pair_ids=pair_ids,
                teacher_raw=teacher_raw,
                gt=gt,
                pose_target=pose_target,
                posenet=posenet,
                segnet=segnet,
            )
            rows.extend(checkpoint["pair_rows"])
            chunk_facts.append(
                {
                    "stem": f"pairs_{pair_ids[0]:04d}_{pair_ids[-1]:04d}",
                    "payload": checkpoint["payload"],
                    "camera_bytes_sha256": checkpoint["camera_bytes_sha256"],
                    "resumed": checkpoint["resumed"],
                }
            )
            print(
                json.dumps({"rung": rung, "realized_pairs": pair_ids[-1] + 1, "n": N, "resumed": checkpoint["resumed"]}),
                flush=True,
            )
        components = aggregate_rung(rows)
        results[rung] = {
            "specification": rung_specification(rung),
            "components": components,
            "chunks": chunk_facts,
        }
        qbt1.atomic_json(output / "checkpoints" / f"stage_02a_{rung}_rung.json", results[rung])
        print(json.dumps({"rung": rung, "components": components}), flush=True)

    receipt = {
        "schema": SCHEMA_STAGE2A,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "axis": "[macOS-CPU advisory]",
        "score_claim": False,
        "promotable": False,
        "research_only": True,
        "lane_id": LANE_ID,
        "lane_claim": claim,
        "verdict_scope": (
            "MEASUREMENT: teacher ceiling and deterministic teacher-corruption ladder on n600 through the "
            "frozen CPU scorer. Object-free: no OBX2 candidate is built, trained, or admitted here."
        ),
        "declared_lever": (
            "Stage 2a is a declared addition to the sealed stage list, inserted under the CLAUDE.md "
            "Carmack MVP-first rule to price the spec's own distortion<0.04 gate before the training burn."
        ),
        "n": N,
        "pair_denominator": N,
        "pixel_denominator": N * EVAL_H * EVAL_W,
        "pose_value_denominator": N * 6,
        "gates": {
            "packet_byte_gate": PACKET_BYTE_GATE,
            "distortion_gate": DISTORTION_GATE,
            "rate_at_packet_byte_gate": RATE_AT_GATE,
            "distortion_budget_for_sub_012": DISTORTION_BUDGET_FOR_SUB_012,
        },
        "pointer_authority_control": {
            "axis": "[contest-CUDA T4 n600]",
            "d_seg": POINTER_AUTHORITY_D_SEG,
            "d_pose": POINTER_AUTHORITY_D_POSE,
            "distortion": 100.0 * POINTER_AUTHORITY_D_SEG + math.sqrt(10.0 * POINTER_AUTHORITY_D_POSE),
            "note": "authority row for the same decoded bytes; never mixed into an advisory row",
        },
        "rungs": results,
        "host": {"platform": platform.platform(), "python": platform.python_version()},
    }
    qbt1.atomic_json(output / "STAGE_2A_RESULT.json", receipt)
    return receipt


# Ordered so the decisive rungs measure first.  `teacher` is the ceiling control,
# `sp_874x1164` proves the back-projection path is faithful on the identity grid,
# and `sp_384x512` is the achievability bound for a renderer on the QBF grid.
DEFAULT_RUNGS = (
    "teacher",
    "sp_874x1164",
    "sp_384x512",
    "grid_384x512",
    "sp_192x256",
    "sp_640x852",
    "sp384_render_noise_1",
    "sp384_render_noise_2",
    "sp384_render_noise_4",
    "sp384_render_noise_8",
    "noise_1",
    "noise_2",
    "noise_4",
    "noise_8",
    "interior_noise_16",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OBX2 edge-local implicit correction stage runner")
    parser.add_argument("stage", choices=("stage0", "stage2a"))
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--claim-id", default=LANE_ID)
    parser.add_argument("--rungs", nargs="*", default=list(DEFAULT_RUNGS))
    parser.add_argument("--launch-authorized", action="store_true")
    parser.add_argument("--resume-from", type=Path, default=OUTPUT_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output
    if args.resume_from.resolve() != output.resolve():
        raise OBX2Error("--resume-from must name the exact OBX2 output root")
    for rung in args.rungs:
        rung_specification(rung)
    if args.stage == "stage0":
        receipt = stage0(output, rungs=args.rungs)
        print(json.dumps({"stage": "stage0", "status": "PASS", "free_bytes": receipt["storage_preflight"]["free_bytes"]}))
        return 0
    if not args.launch_authorized:
        raise OBX2Error("Stage 2a n600 realization requires explicit --launch-authorized")
    stage0(output, rungs=args.rungs)
    receipt = stage2a(output, rungs=args.rungs, claim_id=args.claim_id)
    summary = {
        rung: {
            "distortion": value["components"]["distortion"],
            "d_seg": value["components"]["d_seg"],
            "d_pose": value["components"]["d_pose"],
            "camera_rmse": value["components"]["camera_rmse_vs_teacher"],
            "scorer_plane_rmse": value["components"]["scorer_plane_rmse_vs_teacher"],
            "passes": value["components"]["passes_distortion_gate"],
        }
        for rung, value in receipt["rungs"].items()
    }
    print(json.dumps({"stage": "stage2a", "summary": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
