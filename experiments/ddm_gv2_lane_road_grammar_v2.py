#!/usr/bin/env python3
"""Build the GV2 Road<->Lane segment-event store on the CP135 receiver.

GV2 keeps the EC1 proposal wire format so the landed VD1 validator consumes
the resulting 200-event store unchanged.  The proposal alphabet is different:
each event is a connected, sign-aware Road/Lane boundary segment anchored at a
real CP135 Road<->Lane error.  Two connected supports are constructed at every
anchor/scale and the retained JS4 per-pair PoseNet Jacobian selects the support
with the smaller non-negative first-order pose damage before scorer results
exist.  This is a discrete pose-sensitive-subspace minimization, not the exact
continuous JS4 projection.  A frozen CPU-torch affected-pair pass then supplies
advisory ranking and an honest nonlinear pose-stack prediction.

Every materialized event payload, receiver camera, scorer input, PoseNet input,
logit field, argmax field, and pose output is retained below ``--output``.
The runner is pair-checkpointed and safe to resume by rerunning the same command.
It never launches Modal and never claims an exact score.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import lzma
import math
import os
import shutil
import sys
import time
from heapq import heappop, heappush
from pathlib import Path
from typing import Any, Final

import brotli
import numpy as np
from scipy import ndimage

REPO: Final = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments import ddm_ec1_event_coordinate_producer as ec1
from experiments import ddm_js2b_edge_conditioning_relative_gauge as js2b

DEFAULT_OUTPUT: Final = Path("/Volumes/VertigoDataTier/pact/ddm_gv2_20260812")
DEFAULT_EVENT_STORE: Final = DEFAULT_OUTPUT / "event_store_target_anchored_v2"
PROJECTOR_MANIFEST: Final = Path(
    "/Volumes/VertigoDataTier/pact/ddm_js4_20260812/projector/MANIFEST.json"
)
JO1_ANALYSIS: Final = Path("/Volumes/VertigoDataTier/pact/ddm_jo1_20260812/10_ANALYSIS.json")
BASE_ARGMAX: Final = Path(
    "/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/stage0_per_edge/"
    "candidates/cp135_base/scorer/argmax_n600.npy"
)
GT_ARGMAX: Final = Path(
    "/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/stage0_per_edge/"
    "custody/gt_argmax_n600.npy"
)

N: Final = 600
H: Final = 384
W: Final = 512
PIXELS: Final = N * H * W
STORE_EVENTS: Final = 200
UNITS_PER_PAIR: Final = 8
SUPPORT_SIZES: Final = (12, 24, 48, 96)
VARIANT_ANGLES: Final = (0.0, math.pi)
TARGET_MAX_DISTANCE: Final = 3.0
POSE_STACK_BUDGET: Final = 1.3e-7
OPTIMISTIC_BAR_S: Final = 0.000216
STRETCH_BAR_S: Final = 0.001
CP135_DPOSE: Final = 0.00000688438922225032
RATE_DENOMINATOR: Final = 37_545_489
MIN_FREE_BYTES: Final = 8 * 1024**3
AXIS: Final = "[macOS-CPU advisory, frozen scorer, affected-pair n600 projection]"
SOURCE_ARCHIVE_SHA256: Final = js2b.BASE_ARCHIVE_SHA256
EVENT_TYPE_ID: Final = ec1.EVENT_TYPE["lane_program_delta"]
CONNECT8: Final = np.ones((3, 3), dtype=bool)
BOUNDARY_DILATION: Final = np.ones((3, 3), dtype=bool)


class GV2Error(RuntimeError):
    """A custody, construction, retention, resume, or store invariant failed."""


@dataclasses.dataclass(frozen=True, slots=True)
class Unit:
    pair: int
    source_class: int
    target_class: int
    support_size: int
    anchor_y: int
    anchor_x: int
    ordinal_in_pair: int


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def atomic_bytes(path: Path, payload: bytes) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.{os.getpid()}.partial")
    with partial.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, path)
    return file_record(path)


def atomic_json(path: Path, value: Any) -> dict[str, Any]:
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    return atomic_bytes(path, payload)


def atomic_npy(path: Path, value: np.ndarray) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.{os.getpid()}.partial")
    with partial.open("wb") as stream:
        np.save(stream, np.asarray(value), allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, path)
    return file_record(path)


def require_record(record: dict[str, Any], *, beneath: Path | None = None) -> Path:
    path = Path(str(record["path"])).resolve()
    if beneath is not None:
        try:
            path.relative_to(beneath.resolve())
        except ValueError as exc:
            raise GV2Error(f"retained artifact escapes {beneath}: {path}") from exc
    if not path.is_file() or file_record(path) != record:
        raise GV2Error(f"retained artifact differs: {path}")
    return path


def preflight(output: Path, event_store: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    event_store.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(output)
    if usage.free < MIN_FREE_BYTES:
        raise GV2Error(f"storage preflight failed: {usage.free} < {MIN_FREE_BYTES}")
    required = (
        PROJECTOR_MANIFEST,
        BASE_ARGMAX,
        GT_ARGMAX,
        js2b.BASE_ARCHIVE,
        js2b.BASE_TOKENS,
        js2b.BASE_RAW,
        js2b.CUSTODY_SEG,
        js2b.CUSTODY_POSE,
        js2b.CUSTODY_ARGMAX,
        js2b.GT_CACHE,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise GV2Error(f"missing custody inputs: {missing}")
    if file_record(js2b.BASE_ARCHIVE) != {
        "path": str(js2b.BASE_ARCHIVE.resolve()),
        "bytes": js2b.BASE_ARCHIVE_BYTES,
        "sha256": SOURCE_ARCHIVE_SHA256,
    }:
        raise GV2Error("CP135 archive differs from the charter pin")
    manifest = json.loads(PROJECTOR_MANIFEST.read_text())
    if (
        manifest.get("schema") != "ddm_js4_pose_projector_cache.v1"
        or len(manifest.get("sample", [])) != 32
        or manifest.get("base_archive_sha256") != SOURCE_ARCHIVE_SHA256
    ):
        raise GV2Error("JS4 projector manifest differs from the charter pin")
    result = {
        "schema": "ddm_gv2_preflight.v1",
        "axis": AXIS,
        "score_claim": False,
        "storage": {"free_bytes": usage.free, "required_bytes": MIN_FREE_BYTES},
        "output": str(output.resolve()),
        "event_store": str(event_store.resolve()),
        "inputs": {path.name: file_record(path) for path in required},
        "sample": manifest["sample"],
        "sample_weights": manifest["sample_weights"],
    }
    atomic_json(output / "00_PREFLIGHT.json", result)
    return result


def load_projector_rows() -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    manifest = json.loads(PROJECTOR_MANIFEST.read_text())
    rows: dict[int, dict[str, Any]] = {}
    for row in manifest["pairs"]:
        pair = int(row["pair_id"])
        jacobian_path = require_record(row["jacobian"])
        receipt_path = require_record(row["receipt"])
        receipt = json.loads(receipt_path.read_text())
        if int(receipt["pair_id"]) != pair or int(row["rank"]) != 6:
            raise GV2Error(f"JS4 projector row differs for pair {pair}")
        rows[pair] = {**row, "jacobian_path": str(jacobian_path)}
    if list(rows) != [int(value) for value in manifest["sample"]]:
        raise GV2Error("JS4 projector order differs")
    return manifest, rows


def load_semantic(context: Any) -> Any:
    torch = context.modules.torch
    semantic = context.modules.renderer_runtime.SemanticTokenRenderer(96)
    state = {
        record.schema.name: torch.from_numpy(np.ascontiguousarray(record.values, dtype=np.float32))
        for record in context.records
    }
    semantic.load_state_dict(state, strict=True)
    return semantic.eval().cpu()


def render(semantic: Any, torch: Any, functional: Any, tokens: np.ndarray, pair: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with torch.inference_mode():
        pre_r = semantic(
            torch.from_numpy(np.array(tokens, dtype=np.uint8, copy=True))[None].long(),
            torch.tensor([pair], dtype=torch.long),
        )
        camera = (
            functional.interpolate(
                pre_r,
                size=(js2b.CAMERA_H, js2b.CAMERA_W),
                mode="bilinear",
                align_corners=False,
            )
            .clamp(0.0, 255.0)
            .round()
            .to(torch.uint8)
        )
        scorer = functional.interpolate(camera.float(), size=(H, W), mode="bilinear", align_corners=False)
    return (
        pre_r[0].cpu().numpy().astype(np.float32, copy=False),
        camera[0].permute(1, 2, 0).cpu().numpy(),
        scorer[0].half().cpu().numpy(),
    )


def nearest_boundary_seed(boundary: np.ndarray, anchor_y: int, anchor_x: int, angle: float) -> tuple[int, int]:
    ys, xs = np.nonzero(boundary)
    if not ys.size:
        raise GV2Error("Road/Lane directed boundary has no source sites")
    dy = ys.astype(np.float64) - anchor_y
    dx = xs.astype(np.float64) - anchor_x
    distance = dy * dy + dx * dx
    direction = dy * math.sin(angle) + dx * math.cos(angle)
    order = np.lexsort((xs, ys, -direction, distance))
    pick = int(order[0])
    return int(ys[pick]), int(xs[pick])


def connected_segment(
    boundary: np.ndarray,
    anchor_y: int,
    anchor_x: int,
    size: int,
    angle: float,
) -> np.ndarray:
    """Grow one connected boundary support with a deterministic directional tie-break."""
    component_labels, _ = ndimage.label(boundary, structure=CONNECT8)
    sizes = np.bincount(component_labels.reshape(-1))
    eligible_labels = np.flatnonzero(sizes >= size)
    eligible_labels = eligible_labels[eligible_labels != 0]
    if not len(eligible_labels):
        raise GV2Error(f"no directed boundary component has the required {size} sites")
    eligible = np.isin(component_labels, eligible_labels)
    seed = nearest_boundary_seed(eligible, anchor_y, anchor_x, angle)
    component = component_labels == component_labels[seed]
    if int(component.sum()) < size:
        raise GV2Error(f"boundary component has {int(component.sum())} sites, needs {size}")
    chosen: set[tuple[int, int]] = set()
    queued = {seed}
    queue: list[tuple[float, int, int]] = []

    def priority(y: int, x: int) -> float:
        dy = float(y - anchor_y)
        dx = float(x - anchor_x)
        directional = dy * math.sin(angle) + dx * math.cos(angle)
        return math.hypot(dy, dx) - 0.18 * directional

    heappush(queue, (priority(*seed), seed[0], seed[1]))
    while queue and len(chosen) < size:
        _cost, y, x = heappop(queue)
        if (y, x) in chosen:
            continue
        chosen.add((y, x))
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                site = (ny, nx)
                if (
                    0 <= ny < H
                    and 0 <= nx < W
                    and component[ny, nx]
                    and site not in queued
                ):
                    queued.add(site)
                    heappush(queue, (priority(ny, nx), ny, nx))
    if len(chosen) != size:
        raise GV2Error(f"connected support stopped at {len(chosen)} of {size}")
    indices = np.asarray(sorted(y * W + x for y, x in chosen), dtype=np.int64)
    support = np.zeros((H, W), dtype=bool)
    support.reshape(-1)[indices] = True
    _labels, count = ndimage.label(support, structure=CONNECT8)
    if count != 1 or not np.all(boundary.reshape(-1)[indices]):
        raise GV2Error("constructed support is not one directed boundary segment")
    return indices


def pair_units(
    pair: int,
    tokens: np.ndarray,
    base_argmax: np.ndarray,
    gt_argmax: np.ndarray,
) -> list[Unit]:
    units: list[Unit] = []
    for source, target in ((0, 1), (1, 0)):
        errors = np.argwhere((base_argmax == source) & (gt_argmax == target))
        if not len(errors):
            raise GV2Error(f"pair {pair} direction {source}->{target} has no error anchor")
        boundary = (tokens == source) & ndimage.binary_dilation(
            tokens == target,
            structure=BOUNDARY_DILATION,
        )
        component_labels, _ = ndimage.label(boundary, structure=CONNECT8)
        component_sizes = np.bincount(component_labels.reshape(-1))
        for support_size in SUPPORT_SIZES:
            eligible_labels = np.flatnonzero(component_sizes >= support_size)
            eligible_labels = eligible_labels[eligible_labels != 0]
            if not len(eligible_labels):
                continue
            eligible = np.isin(component_labels, eligible_labels)
            distance = ndimage.distance_transform_edt(~eligible)
            error_distances = distance[errors[:, 0], errors[:, 1]]
            pick = int(
                np.lexsort((errors[:, 1], errors[:, 0], error_distances))[0]
            )
            if float(error_distances[pick]) > TARGET_MAX_DISTANCE:
                continue
            y, x = (int(value) for value in errors[pick])
            units.append(
                Unit(
                    pair=pair,
                    source_class=source,
                    target_class=target,
                    support_size=support_size,
                    anchor_y=y,
                    anchor_x=x,
                    ordinal_in_pair=len(units),
                )
            )
    if len(units) < UNITS_PER_PAIR - 2:
        raise GV2Error(f"pair {pair} produced only {len(units)} target-reachable units")
    return units


def linear_pose_prediction(
    jacobian: np.ndarray,
    correction: np.ndarray,
    base_pose: np.ndarray,
    gt_pose: np.ndarray,
) -> dict[str, Any]:
    flat = np.asarray(correction, dtype=np.float32).reshape(-1)
    shift = np.asarray(jacobian, dtype=np.float32) @ flat
    predicted_pose = np.asarray(base_pose, dtype=np.float64) + shift.astype(np.float64)
    base_error = float(np.mean((np.asarray(base_pose, dtype=np.float64) - gt_pose) ** 2))
    predicted_error = float(np.mean((predicted_pose - gt_pose) ** 2))
    delta_pair = predicted_error - base_error
    return {
        "pose_shift6": [float(value) for value in shift],
        "pose_shift_l2": float(np.linalg.norm(shift.astype(np.float64))),
        "base_d_pose_pair": base_error,
        "predicted_d_pose_pair": predicted_error,
        "predicted_delta_d_pose_pair": delta_pair,
        "predicted_delta_d_pose_global_n600": delta_pair / N,
        "predicted_nonnegative_pose_bound_global_n600": max(0.0, delta_pair / N),
    }


def support_prior_metrics(
    indices: np.ndarray,
    source: int,
    target: int,
    base_argmax: np.ndarray,
    gt_argmax: np.ndarray,
) -> dict[str, Any]:
    support = np.zeros((H, W), dtype=bool)
    support.reshape(-1)[indices] = True
    influence = ndimage.binary_dilation(support, structure=np.ones((7, 7), dtype=bool))
    target_errors = (base_argmax == source) & (gt_argmax == target)
    overclaim = (base_argmax == source) & (gt_argmax == source)
    reach = int(np.count_nonzero(influence & target_errors))
    risk = int(np.count_nonzero(influence & overclaim))
    return {
        "anchor_target_error_in_support_influence": reach > 0,
        "target_error_pixels_in_r3_influence": reach,
        "already_correct_source_pixels_in_r3_influence": risk,
        "sign_precision_prior": reach / (reach + risk) if reach + risk else 0.0,
        "bbox_y0": int(indices.min() // W),
        "bbox_y1": int(indices.max() // W) + 1,
        "bbox_x0": int(np.min(indices % W)),
        "bbox_x1": int(np.max(indices % W)) + 1,
    }


def variant_result(
    context: Any,
    semantic: Any,
    unit: Unit,
    variant: int,
    indices: np.ndarray,
    base_pre_r: np.ndarray,
    jacobian: np.ndarray,
    pair_root: Path,
    base_argmax: np.ndarray,
    gt_argmax: np.ndarray,
) -> dict[str, Any]:
    root = pair_root / "construction_variants" / f"unit_{unit.ordinal_in_pair:02d}_v{variant}"
    result_path = root / "RESULT.json"
    if result_path.is_file():
        prior = json.loads(result_path.read_text())
        for record in prior["retained"].values():
            require_record(record, beneath=pair_root)
        return prior
    candidate = np.asarray(context.tokens[unit.pair]).copy()
    if np.any(candidate.reshape(-1)[indices] != unit.source_class):
        raise GV2Error("directed support source-class precondition differs")
    candidate.reshape(-1)[indices] = unit.target_class
    payload = ec1.proposal_payload(
        unit.pair,
        unit.source_class,
        unit.target_class,
        indices,
        EVENT_TYPE_ID,
    )
    decoded = ec1.decode_proposal(payload)
    if not np.array_equal(decoded[4], indices):
        raise GV2Error("EC1 proposal parse-back differs")
    pre_r, camera, scorer = render(
        semantic,
        context.modules.torch,
        context.modules.functional,
        candidate,
        unit.pair,
    )
    correction = pre_r - base_pre_r
    linear = linear_pose_prediction(
        jacobian,
        correction,
        context.base_pose_output[int(np.flatnonzero(context.sample == unit.pair)[0])],
        np.asarray(context.gt_poses[unit.pair], dtype=np.float64),
    )
    event_br = brotli.compress(payload, quality=11)
    event_xz = lzma.compress(payload, preset=9 | lzma.PRESET_EXTREME)
    retained = {
        "event.ec1p": atomic_bytes(root / "event.ec1p", payload),
        "event.ec1p.br": atomic_bytes(root / "event.ec1p.br", event_br),
        "event.ec1p.xz": atomic_bytes(root / "event.ec1p.xz", event_xz),
        "indices.int64.npy": atomic_npy(root / "indices.int64.npy", indices),
        "candidate_tokens.uint8.npy": atomic_npy(root / "candidate_tokens.uint8.npy", candidate),
        "pre_r.float16.npy": atomic_npy(root / "pre_r.float16.npy", pre_r.astype(np.float16)),
        "correction.float16.npy": atomic_npy(root / "correction.float16.npy", correction.astype(np.float16)),
        "camera.uint8.npy": atomic_npy(root / "camera.uint8.npy", camera),
        "scorer_input.float16.npy": atomic_npy(root / "scorer_input.float16.npy", scorer),
    }
    result = {
        "schema": "ddm_gv2_construction_variant.v1",
        "axis": AXIS,
        "score_claim": False,
        "pair": unit.pair,
        "source_class": unit.source_class,
        "target_class": unit.target_class,
        "event_type": "lane_program_delta",
        "unit": dataclasses.asdict(unit),
        "variant": variant,
        "site_count": len(indices),
        "construction": "connected directed Road/Lane boundary segment, GT-error anchored",
        "prior": support_prior_metrics(indices, unit.source_class, unit.target_class, base_argmax, gt_argmax),
        "js4_linear_pose": linear,
        "retained": retained,
    }
    atomic_json(result_path, result)
    return result


def score_winners(context: Any, unit_rows: list[dict[str, Any]], pair_root: Path) -> list[dict[str, Any]]:
    torch = context.modules.torch
    pair = int(unit_rows[0]["pair"])
    slot = int(np.flatnonzero(context.sample == pair)[0])
    cameras = np.stack(
        [np.load(row["retained"]["camera.uint8.npy"]["path"], allow_pickle=False) for row in unit_rows]
    )
    candidate_seg_input = js2b.preprocess_seg(context.modules.functional, cameras)
    custody_seg = torch.from_numpy(np.asarray(context.custody_seg[pair : pair + 1]).copy()).float()
    transported_seg = custody_seg + (candidate_seg_input - context.base_seg_input[slot : slot + 1])
    logits = js2b.score_seg(context.segnet, transported_seg, batch=len(unit_rows))
    predictions = logits.argmax(axis=1).astype(np.uint8)

    frame0 = np.repeat(context.base_pairs[slot : slot + 1, 0], len(unit_rows), axis=0)
    pairs = np.stack((frame0, cameras), axis=1)
    candidate_pose_input = js2b.preprocess_pairs(context.posenet, pairs, batch=len(unit_rows))
    custody_pose = torch.from_numpy(np.asarray(context.custody_pose[pair : pair + 1]).copy()).float()
    transported_pose = custody_pose + (candidate_pose_input - context.base_pose_input[slot : slot + 1])
    pose_output = js2b.score_pose(context.posenet, transported_pose, batch=len(unit_rows))

    gt = np.asarray(context.gt_labels[pair])
    base_prediction = np.asarray(context.custody_argmax[pair])
    base_wrong = int(np.count_nonzero(base_prediction != gt))
    gt_pose = np.asarray(context.gt_poses[pair], dtype=np.float64)
    base_pose_error = float(context.base_pose_errors[slot])
    output: list[dict[str, Any]] = []
    for index, construction in enumerate(unit_rows):
        candidate_wrong = int(np.count_nonzero(predictions[index] != gt))
        net_flip_gain = base_wrong - candidate_wrong
        candidate_pose_error = float(np.mean((pose_output[index].astype(np.float64) - gt_pose) ** 2))
        nonlinear_delta_global = (candidate_pose_error - base_pose_error) / N
        linear_bound = float(
            construction["js4_linear_pose"]["predicted_nonnegative_pose_bound_global_n600"]
        )
        predicted_pose_bound = max(0.0, linear_bound, nonlinear_delta_global)
        seg_gain = 100.0 * net_flip_gain / PIXELS
        pose_penalty = math.sqrt(10.0 * (CP135_DPOSE + predicted_pose_bound)) - math.sqrt(10.0 * CP135_DPOSE)
        coded_bytes = int(construction["retained"]["event.ec1p.br"]["bytes"])
        standalone_rate_penalty = 25.0 * coded_bytes / RATE_DENOMINATOR
        predicted_net_gain = seg_gain - pose_penalty - standalone_rate_penalty
        root = pair_root / "winners" / f"unit_{int(construction['unit']['ordinal_in_pair']):02d}"
        retained = dict(construction["retained"])
        retained.update(
            {
                "pose_input.float16.npy": atomic_npy(
                    root / "pose_input.float16.npy",
                    candidate_pose_input[index].half().cpu().numpy(),
                ),
                "seg_logits.float32.npy": atomic_npy(root / "seg_logits.float32.npy", logits[index]),
                "seg_argmax.uint8.npy": atomic_npy(root / "seg_argmax.uint8.npy", predictions[index]),
                "pose_output6.float32.npy": atomic_npy(root / "pose_output6.float32.npy", pose_output[index]),
            }
        )
        scored = {
            **construction,
            "schema": "ddm_gv2_scored_candidate.v1",
            "prescreen": {
                "axis": AXIS,
                "base_wrong_pixels_pair": base_wrong,
                "candidate_wrong_pixels_pair": candidate_wrong,
                "net_flip_gain": net_flip_gain,
                "predicted_seg_score_gain_n600": seg_gain,
                "base_d_pose_pair": base_pose_error,
                "candidate_d_pose_pair": candidate_pose_error,
                "nonlinear_cpu_delta_d_pose_global_n600": nonlinear_delta_global,
                "predicted_pose_bound_global_n600": predicted_pose_bound,
                "pose_bound_scope": (
                    "maximum of zero, JS4 first-order prediction, and frozen CPU nonlinear affected-pair "
                    "measurement; advisory prediction, not a CUDA/formal upper bound"
                ),
                "predicted_pose_score_penalty": pose_penalty,
                "standalone_event_brotli_q11_bytes": coded_bytes,
                "standalone_rate_penalty": standalone_rate_penalty,
                "predicted_net_score_gain_standalone_bytes": predicted_net_gain,
            },
            "retained": retained,
        }
        atomic_json(root / "RESULT.json", scored)
        output.append(scored)
    return output


def process_pair(
    context: Any,
    semantic: Any,
    projector: dict[str, Any],
    pair: int,
    output: Path,
    base_argmax: np.ndarray,
    gt_argmax: np.ndarray,
) -> list[dict[str, Any]]:
    pair_root = output / "retained_target_anchored_v2" / f"pair_{pair:06d}"
    result_path = pair_root / "PAIR_RESULT.json"
    if result_path.is_file():
        prior = json.loads(result_path.read_text())
        rows = []
        for record in prior["scored_candidate_receipts"]:
            path = require_record(record, beneath=output)
            rows.append(json.loads(path.read_text()))
        return rows
    torch = context.modules.torch
    functional = context.modules.functional
    base_tokens = np.asarray(context.tokens[pair])
    base_pre_r, base_camera, base_scorer = render(semantic, torch, functional, base_tokens, pair)
    expected_camera = np.asarray(context.raw[2 * pair + 1])
    if not np.array_equal(base_camera, expected_camera):
        raise GV2Error(f"pair {pair} inactive receiver replay differs")
    atomic_npy(pair_root / "base/pre_r.float16.npy", base_pre_r.astype(np.float16))
    atomic_npy(pair_root / "base/camera.uint8.npy", base_camera)
    atomic_npy(pair_root / "base/scorer_input.float16.npy", base_scorer)
    jacobian = np.load(projector["jacobian_path"], mmap_mode="r", allow_pickle=False)
    if jacobian.shape != (6, 3 * H * W) or jacobian.dtype != np.float32:
        raise GV2Error(f"pair {pair} JS4 Jacobian geometry differs")
    units = pair_units(pair, base_tokens, base_argmax, gt_argmax)
    winners: list[dict[str, Any]] = []
    for unit in units:
        boundary = (base_tokens == unit.source_class) & ndimage.binary_dilation(
            base_tokens == unit.target_class,
            structure=BOUNDARY_DILATION,
        )
        variants = []
        seen: set[bytes] = set()
        for variant, angle in enumerate(VARIANT_ANGLES):
            indices = connected_segment(
                boundary,
                unit.anchor_y,
                unit.anchor_x,
                unit.support_size,
                angle,
            )
            identity = indices.tobytes()
            if identity in seen:
                continue
            seen.add(identity)
            variants.append(
                variant_result(
                    context,
                    semantic,
                    unit,
                    variant,
                    indices,
                    base_pre_r,
                    np.asarray(jacobian),
                    pair_root,
                    base_argmax,
                    gt_argmax,
                )
            )
        if not variants:
            raise GV2Error(f"pair {pair} unit {unit.ordinal_in_pair} has no constructed variant")
        winner = min(
            variants,
            key=lambda row: (
                float(row["js4_linear_pose"]["predicted_nonnegative_pose_bound_global_n600"]),
                float(row["js4_linear_pose"]["pose_shift_l2"]),
                -int(row["prior"]["target_error_pixels_in_r3_influence"]),
                int(row["variant"]),
            ),
        )
        winner = {
            **winner,
            "construction_variant_count": len(variants),
            "construction_variant_selection": (
                "min nonnegative JS4 first-order global d_pose, then pose-shift L2, then target-error reach"
            ),
        }
        winners.append(winner)
    scored = score_winners(context, winners, pair_root)
    receipts = []
    for row in scored:
        unit_ordinal = int(row["unit"]["ordinal_in_pair"])
        receipts.append(file_record(pair_root / "winners" / f"unit_{unit_ordinal:02d}" / "RESULT.json"))
    result = {
        "schema": "ddm_gv2_pair_result.v1",
        "axis": AXIS,
        "score_claim": False,
        "pair": pair,
        "candidate_count": len(scored),
        "scored_candidate_receipts": receipts,
    }
    atomic_json(result_path, result)
    return scored


def copy_consumer_payloads(row: dict[str, Any], target_root: Path) -> dict[str, Any]:
    mapping = {
        "event.ec1p": row["retained"]["event.ec1p"],
        "event.ec1p.br": row["retained"]["event.ec1p.br"],
        "event.ec1p.xz": row["retained"]["event.ec1p.xz"],
        "candidate_tokens.uint8.npy": row["retained"]["candidate_tokens.uint8.npy"],
        "camera.uint8.npy": row["retained"]["camera.uint8.npy"],
        "scorer_input.float16.npy": row["retained"]["scorer_input.float16.npy"],
    }
    output = {}
    for name, record in mapping.items():
        source = require_record(record)
        output[name] = atomic_bytes(target_root / name, source.read_bytes())
    return output


def overlap(selected: list[dict[str, Any]], candidate: dict[str, Any]) -> bool:
    candidate_pair = int(candidate["pair"])
    candidate_retained = candidate.get("retained", candidate.get("producer_retained"))
    if not isinstance(candidate_retained, dict):
        raise GV2Error("candidate lacks producer-retained support coordinates")
    candidate_indices = set(
        np.load(candidate_retained["indices.int64.npy"]["path"], allow_pickle=False).tolist()
    )
    for row in selected:
        if int(row["pair"]) != candidate_pair:
            continue
        retained = row.get("retained", row.get("producer_retained"))
        if not isinstance(retained, dict):
            raise GV2Error("selected row lacks producer-retained support coordinates")
        indices = set(np.load(retained["indices.int64.npy"]["path"], allow_pickle=False).tolist())
        if candidate_indices & indices:
            return True
    return False


def intended_selection(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    pose_used = 0.0
    for row in rows:
        gain = int(row["prescreen"]["net_flip_gain"])
        pose_bound = float(row["prescreen"]["predicted_pose_bound_global_n600"])
        if gain <= 0 or pose_used + pose_bound > POSE_STACK_BUDGET or overlap(selected, row):
            continue
        selected.append(row)
        pose_used += pose_bound
    flips = sum(int(row["prescreen"]["net_flip_gain"]) for row in selected)
    optimistic_gain = 100.0 * flips / PIXELS
    return selected, {
        "selection_mode": "greedy positive-gain, support-disjoint, nonnegative pose-bound stack",
        "selected_events": len(selected),
        "selected_proposal_ids": [row["proposal_id"] for row in selected],
        "optimistic_net_flip_gain": flips,
        "optimistic_seg_score_gain": optimistic_gain,
        "pose_stack_prediction_global_n600": pose_used,
        "pose_stack_budget_global_n600": POSE_STACK_BUDGET,
        "pose_stack_within_budget": pose_used <= POSE_STACK_BUDGET,
        "bar_s": OPTIMISTIC_BAR_S,
        "stretch_bar_s": STRETCH_BAR_S,
        "bar_pass": optimistic_gain >= OPTIMISTIC_BAR_S,
        "stretch_pass": optimistic_gain >= STRETCH_BAR_S,
    }


def build_store(
    rows: list[dict[str, Any]],
    event_store: Path,
    sample: list[int],
    context: Any,
) -> dict[str, Any]:
    ranked = sorted(
        rows,
        key=lambda row: (
            -float(row["prescreen"]["predicted_net_score_gain_standalone_bytes"]),
            -int(row["prescreen"]["net_flip_gain"]),
            float(row["prescreen"]["predicted_pose_bound_global_n600"]),
            int(row["pair"]),
            int(row["unit"]["ordinal_in_pair"]),
        ),
    )
    if len(ranked) < STORE_EVENTS:
        raise GV2Error(f"candidate census has {len(ranked)} rows, needs {STORE_EVENTS}")
    emitted = ranked[:STORE_EVENTS]
    index_lines = []
    store_rows = []
    for ordinal, row in enumerate(emitted):
        payload = require_record(row["retained"]["event.ec1p"]).read_bytes()
        proposal_id = f"gv2_{ordinal:04d}_{hashlib.sha256(payload).hexdigest()[:12]}"
        proposal_root = event_store / "proposals" / proposal_id
        consumer_payloads = copy_consumer_payloads(row, proposal_root)
        pair = int(row["pair"])
        slot = int(np.flatnonzero(context.sample == pair)[0])
        candidate_camera = np.load(consumer_payloads["camera.uint8.npy"]["path"], allow_pickle=False)
        candidate_scorer = np.load(
            consumer_payloads["scorer_input.float16.npy"]["path"], allow_pickle=False
        )
        base_camera = np.asarray(context.raw[2 * pair + 1])
        base_scorer = context.base_seg_input[slot].half().cpu().numpy()
        camera_changed = int(np.count_nonzero(candidate_camera != base_camera))
        scorer_changed = int(np.count_nonzero(candidate_scorer != base_scorer))
        if camera_changed == 0 or scorer_changed == 0:
            raise GV2Error(f"ranked proposal {proposal_id} is not receiver-effective")
        receipt = {
            "schema": "ddm_gv2_receiver_proposal.v1",
            "axis": AXIS,
            "score_claim": False,
            "acceptance_tested": False,
            "proposal_id": proposal_id,
            "ordinal": ordinal,
            "pair": pair,
            "event_type": "lane_program_delta",
            "source_class": ec1.CLASSES[int(row["source_class"])],
            "target_class": ec1.CLASSES[int(row["target_class"])],
            "site_count": int(row["site_count"]),
            "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
            "parse_back_exact": True,
            "receiver_effective": True,
            "receiver_effectiveness": {
                "camera_changed_values": camera_changed,
                "scorer_lattice_changed_values": scorer_changed,
                "inactive_base_camera": "CP135 retained exact frame1",
                "inactive_base_scorer": "CP135 retained frame1 through frozen bilinear R",
            },
            "pose_null_construction": (
                "JS4 Jacobian-sensitive discrete support minimization before scorer; actual EC1 token event "
                "is neither the continuous JS4 projection nor an output-space Q3 projection"
            ),
            "exact_pose_null_by_construction_proven": False,
            "predicted_flips": int(row["prescreen"]["net_flip_gain"]),
            "predicted_pose_bound_global_n600": float(
                row["prescreen"]["predicted_pose_bound_global_n600"]
            ),
            "predicted_bytes_brotli_q11": int(consumer_payloads["event.ec1p.br"]["bytes"]),
            "predicted_net_score_gain_standalone_bytes": float(
                row["prescreen"]["predicted_net_score_gain_standalone_bytes"]
            ),
            "construction": {
                "unit": row["unit"],
                "selected_variant": row["variant"],
                "variant_count": row["construction_variant_count"],
                "selection": row["construction_variant_selection"],
                "prior": row["prior"],
                "js4_linear_pose": row["js4_linear_pose"],
            },
            "prescreen": row["prescreen"],
            "producer_retained": row["retained"],
            "consumer_payloads": consumer_payloads,
        }
        receipt_path = proposal_root / "proposal.json"
        atomic_json(receipt_path, receipt)
        indexed = {**receipt, "proposal_receipt": file_record(receipt_path)}
        index_lines.append((json.dumps(indexed, sort_keys=True, allow_nan=False) + "\n").encode())
        store_rows.append(indexed)
    index_record = atomic_bytes(event_store / "proposal_index.jsonl", b"".join(index_lines))
    selected, selection = intended_selection(store_rows)
    state = {
        "schema": "ddm_js5_realized_acceptance_200_store.v1",
        "producer": "ddm_gv2_lane_road_grammar_v2",
        "status": "PRODUCED_NOT_ACCEPTANCE_TESTED",
        "proposal_count": len(store_rows),
        "receiver_effective_count": len(store_rows),
        "attempt_count": sum(int(row["construction_variant_count"]) for row in rows),
        "sample": sample,
        "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
        "proposal_index": index_record,
        "acceptance_tested": False,
        "score_claim": False,
        "axis": AXIS,
        "intended_selection": selection,
        "fire_boundary": "MAIN owns the unchanged VD1 T4 validator and sole n600 scorer slot",
    }
    atomic_json(event_store / "state.json", state)
    return {
        "schema": "ddm_gv2_store_result.v1",
        "event_store": str(event_store.resolve()),
        "state": file_record(event_store / "state.json"),
        "proposal_index": index_record,
        "emitted_events": len(store_rows),
        "candidate_events": len(rows),
        "intended_selection": selection,
        "selected_rows": selected,
    }


def validate_unchanged_vd1(event_store: Path, output: Path) -> dict[str, Any]:
    """Exercise the landed VD1 bundle builder without dispatching its Modal function."""
    from experiments import ddm_vd1_modal_batch_event_validator as vd1

    bundle, manifest = vd1.build_event_bundle(event_store, JO1_ANALYSIS, k=STORE_EVENTS)
    bundle_record = atomic_bytes(output / "unchanged_vd1/event_bundle_k200.zip", bundle)
    if (
        manifest.get("schema") != "ddm_vd1_event_bundle.v1"
        or manifest.get("selection_mode") != "full_200_census"
        or int(manifest.get("selected_events", -1)) != STORE_EVENTS
        or manifest.get("bundle_sha256") != bundle_record["sha256"]
    ):
        raise GV2Error("unchanged VD1 event-bundle receipt differs")
    result = {
        "schema": "ddm_gv2_unchanged_vd1_compatibility.v1",
        "compatible": True,
        "dispatch": False,
        "validator_source": file_record(Path(vd1.__file__).resolve()),
        "validator_worker_source": file_record(
            REPO / "experiments/ddm_vd1_batch_event_validator_worker.py"
        ),
        "jo1_analysis": file_record(JO1_ANALYSIS),
        "event_store": str(event_store.resolve()),
        "bundle": bundle_record,
        "manifest": manifest,
        "axis": "local schema and bundle parse-back only; no scorer",
        "score_claim": False,
    }
    atomic_json(output / "unchanged_vd1/COMPATIBILITY.json", result)
    return result


def write_summary(
    output: Path,
    store: dict[str, Any],
    compatibility: dict[str, Any],
    elapsed: float,
) -> dict[str, Any]:
    selection = store["intended_selection"]
    falsifier = not bool(selection["bar_pass"])
    result = {
        "schema": "ddm_gv2_final_result.v1",
        "axis": AXIS,
        "score_claim": False,
        "pointer_moved": False,
        "store": {key: value for key, value in store.items() if key != "selected_rows"},
        "unchanged_vd1_compatibility": {
            "compatible": compatibility["compatible"],
            "dispatch": compatibility["dispatch"],
            "receipt": file_record(output / "unchanged_vd1/COMPATIBILITY.json"),
            "bundle": compatibility["bundle"],
        },
        "falsifier": {
            "fired": falsifier,
            "bar_s": OPTIMISTIC_BAR_S,
            "measured_optimistic_eligible_gain_s": selection["optimistic_seg_score_gain"],
            "scope": (
                "FORMULATION: CP135 EC1-wire connected Road<->Lane boundary segments of 12/24/48/96 "
                "sites, GT-error anchored, two-support JS4 first-order pose minimization, frozen-CPU "
                "nonlinear pose-stack prediction on the sealed stratified n32 sample"
            ),
            "route_if_fired": "implicit edge conditioning, JS1 stage-0 lineage; no GV3 sparse-event arm",
        },
        "boundaries": {
            "modal_dispatch": False,
            "contest_cuda_validator": False,
            "candidate_archive": False,
            "exact_score": False,
            "q3_output_space_projection_in_wire_format": False,
            "exact_pose_null_event_constructed": False,
            "jo1_plus3_byte_transfer_remeasured": False,
        },
        "wall_seconds": elapsed,
    }
    atomic_json(output / "FINAL_RESULT.json", result)
    return result


def run(output: Path, event_store: Path) -> dict[str, Any]:
    started = time.perf_counter()
    preflight_row = preflight(output, event_store)
    manifest, projector_rows = load_projector_rows()
    context = js2b.build_context(output / "context")
    if context.sample.tolist() != manifest["sample"]:
        raise GV2Error("CPU scorer sample differs from JS4 projector sample")
    semantic = load_semantic(context)
    base_argmax = np.load(BASE_ARGMAX, mmap_mode="r", allow_pickle=False)
    gt_argmax = np.load(GT_ARGMAX, mmap_mode="r", allow_pickle=False)
    if base_argmax.shape != (N, H, W) or gt_argmax.shape != (N, H, W):
        raise GV2Error("argmax custody geometry differs")
    all_rows: list[dict[str, Any]] = []
    for completed, pair in enumerate(context.sample.tolist()):
        pair_rows = process_pair(
            context,
            semantic,
            projector_rows[int(pair)],
            int(pair),
            output,
            np.asarray(base_argmax[int(pair)]),
            np.asarray(gt_argmax[int(pair)]),
        )
        all_rows.extend(pair_rows)
        atomic_json(
            output / "RUN_STATE.json",
            {
                "schema": "ddm_gv2_run_state.v1",
                "status": "RUNNING",
                "resumable": True,
                "completed_pairs": completed + 1,
                "required_pairs": len(context.sample),
                "latest_pair": int(pair),
                "materialized_scored_candidates": len(all_rows),
                "sample": preflight_row["sample"],
                "axis": AXIS,
                "score_claim": False,
            },
        )
    store = build_store(all_rows, event_store, context.sample.tolist(), context)
    compatibility = validate_unchanged_vd1(event_store, output)
    result = write_summary(output, store, compatibility, time.perf_counter() - started)
    atomic_json(
        output / "RUN_STATE.json",
        {
            "schema": "ddm_gv2_run_state.v1",
            "status": "COMPLETE",
            "resumable": True,
            "completed_pairs": len(context.sample),
            "required_pairs": len(context.sample),
            "materialized_scored_candidates": len(all_rows),
            "event_store": str(event_store.resolve()),
            "final_result": file_record(output / "FINAL_RESULT.json"),
            "axis": AXIS,
            "score_claim": False,
        },
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--event-store", type=Path, default=DEFAULT_EVENT_STORE)
    args = parser.parse_args(argv)
    result = run(args.output.resolve(), args.event_store.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
