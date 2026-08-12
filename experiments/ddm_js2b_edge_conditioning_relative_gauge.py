#!/usr/bin/env python3
"""Relative-gauge implicit edge-conditioning screen on the CP135 receiver.

This runner never treats local CPU SegNet values as absolute score evidence.
It screens two-symbol FiLM corrections on a seeded stratified-random n=32
sample, transports each receiver-realized correction onto the retained T4
scorer-input planes, and ranks only the change in local flip count.  The best
semantic seed is coupled to an exhaustive per-row +/-1 int12 carrier descent
until a dry pass, then rebuilt through CP135's real split-Brotli/CAP1 container.

Every compressed payload, rendered seed, scorer argmax/logit field, carrier
checkpoint, and final archive produced by the run is retained below --output.
The full-n600 T4 acceptance row is emitted only when the charter's projected
flip and byte gates pass; this arm never dispatches Modal.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib
import io
import json
import math
import os
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
UPSTREAM = REPO / "upstream"
BOOK = Path("/Volumes/VertigoDataTier/pact/pr135_intake_20260810/experiment_book")
BOOK_SRC = BOOK / "src"
BOOK_SCRIPTS = BOOK / "scripts"
RUNTIME = Path("/Volumes/VertigoDataTier/pact/ddm_cp135_20260810/adapted_runtime")
BASE_ARCHIVE = RUNTIME / "archive.zip"
BASE_ARCHIVE_BYTES = 186_252
BASE_ARCHIVE_SHA256 = "6eb1a3b79cb167e03372339e07e93cae13b6ba3114a9eb917288bb038622edb6"
BASE_RAW = Path(
    "/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/stage0_per_edge/candidates/cp135_base/retained/0.raw"
)
BASE_TOKENS = BASE_RAW.parent / "decoded_tokens_n600.npy"
BASE_LOCAL_LOGITS = BASE_RAW.parent.parent / "scorer/logits_n600.float32.npy"
CUSTODY_ROOT = Path("/Volumes/VertigoDataTier/pact/ddm_js2_20260812/instrument_validation_cuda")
CUSTODY_MANIFEST = CUSTODY_ROOT / "scorer_input_cache_tensors/manifest.json"
CUSTODY_SEG = CUSTODY_ROOT / "scorer_input_cache_tensors/segnet_last_rgb.npy"
CUSTODY_POSE = CUSTODY_ROOT / "scorer_input_cache_tensors/posenet_yuv6_pair.npy"
CUSTODY_ARGMAX = CUSTODY_ROOT / "lstars_local_on_custody.npy"
INSTRUMENT_RECEIPT = CUSTODY_ROOT / "INSTRUMENT_VALIDATION_CUDA_CUSTODY.json"
GT_CACHE = REPO / "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"
DEFAULT_OUTPUT = Path("/Volumes/VertigoDataTier/pact/ddm_js2b_20260812")
BROTLI = Path("/opt/homebrew/bin/brotli")

N = 600
SAMPLE_N = 32
H = 384
W = 512
CAMERA_H = 874
CAMERA_W = 1164
D = 12
CLASSES = 5
LOCAL_BASELINE_FLIPS = 50_389
PROMOTED_SCALAR_FLIPS = 34_968
INSTRUMENT_FLOOR_S = 0.0131
SEED = 20_260_812
BATCH = 16
THREADS = 8
RATE_DENOMINATOR = 37_545_489
POSE_GUARD = 2e-6
T4_FLIP_GATE = -2_000
T4_BYTE_GATE = 1_000
MAX_COMPENSATION_PASSES = 12
AXIS = "[macOS-CPU advisory, instrument floor 0.0131 S]"

FILM = "blocks.3.film.weight"
SEMANTIC_CATALOG: tuple[tuple[str, tuple[tuple[int, int], ...]], ...] = (
    ("f26_pair_continue", ((703, -1), (1514, -1))),
    ("f19_623p_1512m", ((623, 1), (1512, -1))),
    ("f19_623p_1335m", ((623, 1), (1335, -1))),
    ("f19_623p_399p", ((623, 1), (399, 1))),
    ("f19_623p_553p", ((623, 1), (553, 1))),
    ("f19_1512m_1335m", ((1512, -1), (1335, -1))),
    ("f19_623p_701m", ((623, 1), (701, -1))),
    ("f19_623p_1415m", ((623, 1), (1415, -1))),
    ("f19_1512m_399p", ((1512, -1), (399, 1))),
)
MOVES = tuple((dimension, delta) for dimension in range(D) for delta in (-1, 1))


class JS2BError(RuntimeError):
    """A custody, mechanism, scorer, resume, or receiver invariant failed."""


@dataclasses.dataclass(frozen=True)
class Modules:
    torch: Any
    functional: Any
    challenge: Any
    load_file: Any
    runtime: Any
    renderer_runtime: Any
    residual: Any
    carrier: Any
    book_carrier: Any
    frame0_selector: Any
    coefficient_codec: Any
    renderer_codec: Any
    baseline: Any
    segnet_safe: Any
    book_residual: Any
    exact_carrier_renderer: Any
    candidate_cpr1: Any
    cp135: Any


@dataclasses.dataclass
class Context:
    modules: Modules
    parts: Any
    records: tuple[Any, ...]
    payload: Any
    selector: bytes
    selector_choices: np.ndarray
    canonical_carrier: bytes
    codes: np.ndarray
    semantic_renderer: Any
    carrier_renderer: Any
    segnet: Any
    posenet: Any
    tokens: np.ndarray
    raw: np.memmap
    gt_labels: np.ndarray
    gt_poses: np.ndarray
    custody_seg: np.ndarray
    custody_pose: np.ndarray
    custody_argmax: np.ndarray
    sample: np.ndarray
    sample_weights: np.ndarray
    base_pairs: np.ndarray
    base_seg_input: Any
    base_pose_input: Any
    base_pose_output: np.ndarray
    base_pose_errors: np.ndarray


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def atomic_npy(path: Path, value: np.ndarray) -> None:
    buffer = io.BytesIO()
    np.save(buffer, np.ascontiguousarray(value), allow_pickle=False)
    atomic_bytes(path, buffer.getvalue())


def quantiles(values: np.ndarray) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if not array.size:
        return {
            "count": 0,
            "min": None,
            "p10": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    points = np.quantile(array, [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0])
    return {
        "count": int(array.size),
        "min": float(points[0]),
        "p10": float(points[1]),
        "p25": float(points[2]),
        "p50": float(points[3]),
        "p75": float(points[4]),
        "p90": float(points[5]),
        "p95": float(points[6]),
        "p99": float(points[7]),
        "max": float(points[8]),
    }


def stratified_sample(seed: int = SEED, population: int = N, count: int = SAMPLE_N) -> tuple[np.ndarray, np.ndarray]:
    if not 0 < count <= population:
        raise ValueError("sample count must be within the population")
    edges = np.linspace(0, population, count + 1, dtype=np.int64)
    rng = np.random.default_rng(seed)
    sample = np.asarray(
        [rng.integers(edges[index], edges[index + 1]) for index in range(count)],
        dtype=np.int64,
    )
    weights = np.diff(edges).astype(np.int64)
    return sample, weights


def projected_sum(per_pair: np.ndarray, weights: np.ndarray) -> int:
    values = np.asarray(per_pair, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.int64)
    if values.shape != weights.shape:
        raise ValueError("per-pair values and stratum weights differ")
    return int(np.dot(values, weights))


def stratified_mean(per_pair: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(per_pair, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.shape != weights.shape or float(weights.sum()) <= 0.0:
        raise ValueError("per-pair values and stratum weights differ")
    return float(np.dot(values, weights) / weights.sum())


def fire_gate(projected_robust_delta_flips: int, delta_bytes: int, pose_delta: float) -> bool:
    return projected_robust_delta_flips <= T4_FLIP_GATE and delta_bytes <= T4_BYTE_GATE and pose_delta < POSE_GUARD


def load_modules() -> Modules:
    for path in (str(UPSTREAM), str(BOOK_SRC), str(BOOK_SCRIPTS), str(REPO), str(RUNTIME)):
        if path not in sys.path:
            sys.path.insert(0, path)
    import modules as challenge
    import runtime.carrier_repack as carrier
    import runtime.f26_inflate as runtime
    import runtime.residual_archive as residual
    import torch
    from cpr1_sub4 import baseline, frame0_selector, segnet_safe
    from cpr1_sub4 import carrier_repack as book_carrier
    from cpr1_sub4 import residual_archive as book_residual
    from cpr1_sub4.entropy import coefficient_ar1_codec as coefficient_codec
    from cpr1_sub4.entropy import renderer_weight_codec as renderer_codec
    from polish_carrier_coefficients import ExactCarrierRenderer
    from safetensors.torch import load_file
    from solve_f18_pr133_tail import _candidate_cpr1
    from torch.nn import functional

    from experiments import ddm_cp135_rate_compose as cp135

    if Path(challenge.__file__).resolve() != (UPSTREAM / "modules.py").resolve():
        raise JS2BError("loaded a non-custodied upstream scorer module")
    renderer_runtime = runtime._load_renderer(RUNTIME / "cpr1")
    return Modules(
        torch=torch,
        functional=functional,
        challenge=challenge,
        load_file=load_file,
        runtime=runtime,
        renderer_runtime=renderer_runtime,
        residual=residual,
        carrier=carrier,
        book_carrier=book_carrier,
        frame0_selector=frame0_selector,
        coefficient_codec=coefficient_codec,
        renderer_codec=renderer_codec,
        baseline=baseline,
        segnet_safe=segnet_safe,
        book_residual=book_residual,
        exact_carrier_renderer=ExactCarrierRenderer,
        candidate_cpr1=_candidate_cpr1,
        cp135=cp135,
    )


def require_custody() -> dict[str, Any]:
    required = (
        BASE_ARCHIVE,
        BASE_RAW,
        BASE_TOKENS,
        BASE_LOCAL_LOGITS,
        CUSTODY_MANIFEST,
        CUSTODY_SEG,
        CUSTODY_POSE,
        CUSTODY_ARGMAX,
        INSTRUMENT_RECEIPT,
        GT_CACHE,
        BROTLI,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise JS2BError(f"missing custody inputs: {missing}")
    if BASE_ARCHIVE.stat().st_size != BASE_ARCHIVE_BYTES or sha256_file(BASE_ARCHIVE) != BASE_ARCHIVE_SHA256:
        raise JS2BError("CP135 archive differs from the charter pin")
    instrument = json.loads(INSTRUMENT_RECEIPT.read_text())
    manifest = json.loads(CUSTODY_MANIFEST.read_text())
    if (
        int(instrument["flips_local_on_custody"]) != LOCAL_BASELINE_FLIPS
        or int(instrument["flips_promoted_scalar_derived"]) != PROMOTED_SCALAR_FLIPS
        or int(instrument["batch"]) != BATCH
        or manifest["archive_sha256"] != BASE_ARCHIVE_SHA256
    ):
        raise JS2BError("corrected instrument custody does not match the charter")
    return {"instrument": instrument, "manifest": manifest}


def load_model_pair(modules: Modules) -> tuple[Any, Any]:
    torch = modules.torch
    torch.set_num_threads(THREADS)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    segnet = modules.challenge.SegNet().eval().cpu()
    segnet.load_state_dict(modules.load_file(str(UPSTREAM / "models/segnet.safetensors"), device="cpu"))
    posenet = modules.challenge.PoseNet().eval().cpu()
    posenet.load_state_dict(modules.load_file(str(UPSTREAM / "models/posenet.safetensors"), device="cpu"))
    for model in (segnet, posenet):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    return segnet, posenet


def score_pose(posenet: Any, inputs: Any, batch: int = BATCH) -> np.ndarray:
    torch = importlib.import_module("torch")
    rows = []
    with torch.inference_mode():
        for start in range(0, int(inputs.shape[0]), batch):
            rows.append(posenet(inputs[start : start + batch])["pose"][..., :6].cpu().numpy())
    return np.concatenate(rows).astype(np.float32, copy=False)


def preprocess_pairs(posenet: Any, pairs: np.ndarray, batch: int = BATCH) -> Any:
    torch = importlib.import_module("torch")
    rows = []
    with torch.inference_mode():
        for start in range(0, len(pairs), batch):
            value = torch.from_numpy(np.ascontiguousarray(pairs[start : start + batch])).permute(0, 1, 4, 2, 3).float()
            rows.append(posenet.preprocess_input(value).cpu())
    return torch.cat(rows)


def preprocess_seg(functional: Any, frames: np.ndarray) -> Any:
    torch = importlib.import_module("torch")
    value = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2).float()
    return functional.interpolate(value, size=(H, W), mode="bilinear")


def score_seg(segnet: Any, inputs: Any, batch: int = BATCH) -> np.ndarray:
    torch = importlib.import_module("torch")
    rows = []
    with torch.inference_mode():
        for start in range(0, int(inputs.shape[0]), batch):
            rows.append(segnet(inputs[start : start + batch]).cpu().numpy().astype(np.float32, copy=False))
    return np.concatenate(rows)


def render_semantic(context: Context, records: tuple[Any, ...]) -> np.ndarray:
    torch = context.modules.torch
    semantic = context.modules.renderer_runtime.SemanticTokenRenderer(96)
    state = {
        record.schema.name: torch.from_numpy(np.ascontiguousarray(record.values, dtype=np.float32))
        for record in records
    }
    semantic.load_state_dict(state, strict=True)
    semantic.eval()
    tokens = torch.from_numpy(np.asarray(context.tokens[context.sample]).astype(np.int64, copy=True))
    indices = torch.from_numpy(context.sample.copy()).long()
    with torch.inference_mode():
        value = semantic(tokens, indices)
        value = context.modules.functional.interpolate(
            value,
            size=(CAMERA_H, CAMERA_W),
            mode="bilinear",
            align_corners=False,
        )
        value = value.clamp(0.0, 255.0).round().to(torch.uint8)
    return value.permute(0, 2, 3, 1).cpu().numpy()


def build_context(output: Path) -> Context:
    custody = require_custody()
    modules = load_modules()
    parts = modules.residual.read_residual_archive(BASE_ARCHIVE)
    base_carrier, selector = modules.carrier.split_frame0_selector_carrier(parts.carrier_blob)
    if selector is None:
        raise JS2BError("CP135 lost its F0E1 selector")
    canonical = modules.coefficient_codec.decode_cap1(base_carrier, frames=N, dimensions=D)
    codes = modules.book_carrier.decode_cpr1_coefficients(canonical, frames=N, dimensions=D).astype(np.int32)
    _, selector_choices = modules.frame0_selector.decode_selector(selector)
    records = modules.renderer_codec.decode_wans1(parts.semantic_blob)
    legacy = modules.baseline.encode_legacy_w4(records)
    payload = modules.baseline.BaselinePayload(
        archive_path=BASE_ARCHIVE,
        semantic_blob=legacy,
        carrier_blob=parts.carrier_blob,
        hpac_blob=parts.hpac_blob,
        token_stream=parts.token_stream,
        records=records,
    )
    reencoded, _ = modules.renderer_codec.encode_wans1(records, strategy="global")
    if reencoded != parts.semantic_blob:
        raise JS2BError("base F26 semantic does not re-encode byte-identically")
    sample, sample_weights = stratified_sample()
    tokens = np.load(BASE_TOKENS, mmap_mode="r", allow_pickle=False)
    if tokens.shape != (N, H, W) or tokens.dtype != np.uint8:
        raise JS2BError("retained CP135 token geometry differs")
    raw = np.memmap(BASE_RAW, mode="r", dtype=np.uint8, shape=(N * 2, CAMERA_H, CAMERA_W, 3))
    gt = np.load(GT_CACHE, mmap_mode="r", allow_pickle=False)
    gt_labels = np.asarray(gt["lstars"])
    gt_poses = np.asarray(gt["gt_poses"])
    custody_seg = np.load(CUSTODY_SEG, mmap_mode="r", allow_pickle=False)
    custody_pose = np.load(CUSTODY_POSE, mmap_mode="r", allow_pickle=False)
    custody_argmax = np.load(CUSTODY_ARGMAX, mmap_mode="r", allow_pickle=False)
    if (
        gt_labels.shape != (N, H, W)
        or gt_poses.shape != (N, 6)
        or custody_seg.shape != (N, 3, H, W)
        or custody_pose.shape != (N, 12, 192, 256)
        or custody_argmax.shape != (N, H, W)
    ):
        raise JS2BError("custody scorer or GT geometry differs")
    segnet, posenet = load_model_pair(modules)
    base_pairs = (
        np.asarray(raw[(2 * sample[:, None] + np.arange(2)).reshape(-1)])
        .reshape(SAMPLE_N, 2, CAMERA_H, CAMERA_W, 3)
        .copy()
    )
    base_seg_input = preprocess_seg(modules.functional, base_pairs[:, 1])
    base_pose_input = preprocess_pairs(posenet, base_pairs)
    custody_pose_sample = modules.torch.from_numpy(np.asarray(custody_pose[sample]).copy())
    base_pose_output = score_pose(posenet, custody_pose_sample)
    base_pose_errors = ((base_pose_output.astype(np.float64) - gt_poses[sample]) ** 2).mean(axis=1)
    renderer = modules.exact_carrier_renderer.create(modules.renderer_runtime, canonical, modules.torch.device("cpu"))
    semantic_renderer = modules.renderer_runtime.SemanticTokenRenderer(96).eval()
    context = Context(
        modules=modules,
        parts=parts,
        records=records,
        payload=payload,
        selector=selector,
        selector_choices=np.asarray(selector_choices),
        canonical_carrier=canonical,
        codes=codes,
        semantic_renderer=semantic_renderer,
        carrier_renderer=renderer,
        segnet=segnet,
        posenet=posenet,
        tokens=tokens,
        raw=raw,
        gt_labels=gt_labels,
        gt_poses=gt_poses,
        custody_seg=custody_seg,
        custody_pose=custody_pose,
        custody_argmax=custody_argmax,
        sample=sample,
        sample_weights=sample_weights,
        base_pairs=base_pairs,
        base_seg_input=base_seg_input,
        base_pose_input=base_pose_input,
        base_pose_output=base_pose_output,
        base_pose_errors=base_pose_errors,
    )
    atomic_json(
        output / "inputs/CUSTODY.json",
        {
            "schema": "ddm_js2b_custody.v1",
            "base_archive": file_record(BASE_ARCHIVE),
            "base_raw": file_record(BASE_RAW),
            "tokens": file_record(BASE_TOKENS),
            "local_logits_six_flip_equivalent": file_record(BASE_LOCAL_LOGITS),
            "custody_manifest": file_record(CUSTODY_MANIFEST),
            "custody_seg": file_record(CUSTODY_SEG),
            "custody_pose": file_record(CUSTODY_POSE),
            "custody_argmax": file_record(CUSTODY_ARGMAX),
            "gt_cache": file_record(GT_CACHE),
            "instrument_receipt": file_record(INSTRUMENT_RECEIPT),
            "instrument": custody["instrument"],
            "sample": sample.tolist(),
            "sample_weights": sample_weights.tolist(),
            "seed": SEED,
            "batch": BATCH,
            "threads": THREADS,
            "axis": AXIS,
            "score_claim": False,
        },
    )
    return context


def calibrate_delta(context: Context, output: Path) -> dict[str, Any]:
    result_path = output / "calibration/RESULT.json"
    margins_path = output / "calibration/local_error_margins.six_flip_equivalent.npy"
    if result_path.is_file():
        result = json.loads(result_path.read_text())
        if result.get("margins") != file_record(margins_path):
            raise JS2BError("calibration checkpoint lost its retained margins")
        return result
    logits = np.load(BASE_LOCAL_LOGITS, mmap_mode="r", allow_pickle=False)
    if logits.shape != (N, CLASSES, H, W) or logits.dtype != np.float32:
        raise JS2BError("retained local logits geometry differs")
    values: list[np.ndarray] = []
    argmax_disagreements = 0
    error_count = 0
    for start in range(0, N, BATCH):
        stop = min(start + BATCH, N)
        chunk = np.asarray(logits[start:stop])
        predicted = chunk.argmax(axis=1).astype(np.uint8)
        custody = np.asarray(context.custody_argmax[start:stop])
        gt = np.asarray(context.gt_labels[start:stop])
        argmax_disagreements += int(np.count_nonzero(predicted != custody))
        wrong = custody != gt
        error_count += int(wrong.sum())
        top = np.partition(chunk, -2, axis=1)[:, -2:]
        margin = top[:, 1] - top[:, 0]
        values.append(np.asarray(margin[wrong], dtype=np.float32))
    margins = np.concatenate(values)
    if error_count != LOCAL_BASELINE_FLIPS or len(margins) != LOCAL_BASELINE_FLIPS:
        raise JS2BError("local custody error denominator differs from the corrected instrument")
    atomic_npy(margins_path, margins)
    recovered = LOCAL_BASELINE_FLIPS - PROMOTED_SCALAR_FLIPS
    ordered = np.sort(margins)
    delta = float(ordered[recovered - 1])
    result = {
        "schema": "ddm_js2b_margin_calibration.v1",
        "axis": AXIS,
        "score_claim": False,
        "local_baseline_flips": LOCAL_BASELINE_FLIPS,
        "promoted_scalar_flips": PROMOTED_SCALAR_FLIPS,
        "net_scalar_difference_flips": recovered,
        "local_decode_logits_vs_custody_argmax_disagreements": argmax_disagreements,
        "margins": file_record(margins_path),
        "all_local_error_margin_distribution": quantiles(margins),
        "delta": delta,
        "delta_definition": (
            "upper edge of the lowest-margin net-scalar-difference-sized subset of local custody errors"
        ),
        "calibration_scope": "CONSERVATIVE_RANK_BAR_NOT_PIXELWISE_CUDA_DISAGREEMENT",
        "limitation": (
            "the promoted CUDA argmax field was not retained, so scalar arithmetic cannot "
            "identify the actual disagreement pixels; exact pixelwise calibration remains queued"
        ),
        "promoted_argmax_field_present": False,
        "absolute_local_progress_claim": False,
    }
    atomic_json(result_path, result)
    return result


def margin_stats(
    base: np.ndarray,
    candidate: np.ndarray,
    gt: np.ndarray,
    logits: np.ndarray,
    delta: float,
    weights: np.ndarray,
) -> dict[str, Any]:
    top = np.partition(logits, -2, axis=1)[:, -2:]
    margin = top[:, 1] - top[:, 0]
    changed = candidate != base
    base_correct = base == gt
    candidate_correct = candidate == gt
    beneficial = changed & ~base_correct & candidate_correct
    harmful = changed & base_correct & ~candidate_correct
    robust = margin >= delta
    total_delta_per_pair = (candidate != gt).sum(axis=(1, 2)) - (base != gt).sum(axis=(1, 2))
    robust_delta_per_pair = (harmful & robust).sum(axis=(1, 2)) - (beneficial & robust).sum(axis=(1, 2))
    fragile_beneficial = beneficial & ~robust
    return {
        "sample_delta_flips": int(total_delta_per_pair.sum()),
        "projected_n600_delta_flips": projected_sum(total_delta_per_pair, weights),
        "sample_robust_delta_flips": int(robust_delta_per_pair.sum()),
        "projected_n600_robust_delta_flips": projected_sum(robust_delta_per_pair, weights),
        "changed_pixels": int(changed.sum()),
        "beneficial_flips": int(beneficial.sum()),
        "harmful_flips": int(harmful.sum()),
        "robust_beneficial_flips": int((beneficial & robust).sum()),
        "robust_harmful_flips": int((harmful & robust).sum()),
        "tie_fragile_beneficial_flips": int(fragile_beneficial.sum()),
        "tie_fragile_fraction_of_beneficial": (
            float(fragile_beneficial.sum() / beneficial.sum()) if beneficial.any() else None
        ),
        "changed_margin_distribution": quantiles(margin[changed]),
        "beneficial_margin_distribution": quantiles(margin[beneficial]),
        "delta": delta,
    }


def semantic_moves(context: Context, moves: Iterable[tuple[int, int]]) -> tuple[Any, ...]:
    edits = tuple(context.modules.segnet_safe.SymbolMove(FILM, int(index), int(delta)) for index, delta in moves)
    return context.modules.segnet_safe.apply_w4_symbol_moves(context.records, edits)


def screen_semantics(context: Context, output: Path, delta: float) -> dict[str, Any]:
    result_path = output / "semantic_screen/RESULT.json"
    rows = []
    prior: dict[str, Any] | None = None
    if result_path.is_file():
        prior = json.loads(result_path.read_text())
        rows = list(prior.get("rows", []))
        if [row["candidate_id"] for row in rows] != [item[0] for item in SEMANTIC_CATALOG[: len(rows)]]:
            raise JS2BError("semantic screen checkpoint differs from the fixed catalog")
    base_pred = np.asarray(context.custody_argmax[context.sample])
    gt = np.asarray(context.gt_labels[context.sample])
    custody_input = context.modules.torch.from_numpy(np.asarray(context.custody_seg[context.sample]).copy())
    for ordinal, (candidate_id, moves) in enumerate(SEMANTIC_CATALOG, start=1):
        if ordinal <= len(rows):
            continue
        root = output / "semantic_screen/candidates" / candidate_id
        records = semantic_moves(context, moves)
        semantic_blob, codec_report = context.modules.renderer_codec.encode_wans1(records, strategy="global")
        semantic_path = root / "semantic.wans1"
        atomic_bytes(semantic_path, semantic_blob)
        frame1 = render_semantic(context, records)
        frame_path = root / "frame1_n32.uint8.npy"
        atomic_npy(frame_path, frame1)
        candidate_local_input = preprocess_seg(context.modules.functional, frame1)
        transported = custody_input + (candidate_local_input - context.base_seg_input)
        logits = score_seg(context.segnet, transported)
        predicted = logits.argmax(axis=1).astype(np.uint8)
        logits_path = root / "logits_n32.float32.npy"
        argmax_path = root / "argmax_n32.npy"
        atomic_npy(logits_path, logits)
        atomic_npy(argmax_path, predicted)
        stats = margin_stats(base_pred, predicted, gt, logits, delta, context.sample_weights)
        row = {
            "ordinal": ordinal,
            "candidate_id": candidate_id,
            "moves": [{"tensor": FILM, "flat_index": index, "delta": move} for index, move in moves],
            "semantic": file_record(semantic_path),
            "semantic_codec_report": codec_report,
            "frame1": file_record(frame_path),
            "logits": file_record(logits_path),
            "argmax": file_record(argmax_path),
            "metrics": stats,
            "axis": AXIS,
            "score_claim": False,
            "mechanism_scope": "two FiLM codes; carrier compensation pending",
        }
        atomic_json(root / "RESULT.json", row)
        rows.append(row)
        partial = {
            "schema": "ddm_js2b_semantic_screen.v1",
            "complete": ordinal == len(SEMANTIC_CATALOG),
            "axis": AXIS,
            "score_claim": False,
            "catalog_denominator": len(SEMANTIC_CATALOG),
            "completed": ordinal,
            "sample": context.sample.tolist(),
            "sample_weights": context.sample_weights.tolist(),
            "delta": delta,
            "rows": rows,
        }
        atomic_json(result_path, partial)
        print(json.dumps({"stage": "semantic_screen", "candidate": candidate_id, **stats}, sort_keys=True), flush=True)
    ranked = sorted(
        rows,
        key=lambda row: (
            int(row["metrics"]["projected_n600_robust_delta_flips"]),
            int(row["metrics"]["projected_n600_delta_flips"]),
            row["candidate_id"],
        ),
    )
    result = json.loads(result_path.read_text())
    result.update(
        {
            "complete": True,
            "ranking": [row["candidate_id"] for row in ranked],
            "selected_for_carrier_compensation": ranked[0]["candidate_id"],
            "selection": "minimum projected margin-robust delta flips; total delta then id break ties",
        }
    )
    atomic_json(result_path, result)
    return result


def selected_records(context: Context, screen: dict[str, Any]) -> tuple[str, tuple[Any, ...], bytes]:
    candidate_id = str(screen["selected_for_carrier_compensation"])
    moves = dict(SEMANTIC_CATALOG)[candidate_id]
    records = semantic_moves(context, moves)
    blob, _ = context.modules.renderer_codec.encode_wans1(records, strategy="global")
    retained = Path(next(row["semantic"]["path"] for row in screen["rows"] if row["candidate_id"] == candidate_id))
    if file_record(retained)["sha256"] != sha256_bytes(blob):
        raise JS2BError("selected semantic payload differs from its retained screen seed")
    return candidate_id, records, blob


def pose_candidate(
    context: Context,
    candidate_frame1: np.ndarray,
    codes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame0 = context.carrier_renderer.render(codes[context.sample])
    selected = context.selector_choices[context.sample] != 0
    if np.any(selected):
        frame0[selected] = context.base_pairs[selected, 0]
    pairs = np.stack((frame0, candidate_frame1), axis=1)
    candidate_pose_input = preprocess_pairs(context.posenet, pairs)
    custody = context.modules.torch.from_numpy(np.asarray(context.custody_pose[context.sample]).copy())
    transported = custody + (candidate_pose_input - context.base_pose_input)
    outputs = score_pose(context.posenet, transported)
    errors = ((outputs.astype(np.float64) - context.gt_poses[context.sample]) ** 2).mean(axis=1)
    return errors, outputs, frame0


def compensate_carrier(context: Context, output: Path, screen: dict[str, Any]) -> dict[str, Any]:
    result_path = output / "carrier_compensation/RESULT.json"
    candidate_id, records, semantic_blob = selected_records(context, screen)
    frame1_path = Path(next(row["frame1"]["path"] for row in screen["rows"] if row["candidate_id"] == candidate_id))
    candidate_frame1 = np.load(frame1_path, allow_pickle=False)
    checkpoint_root = output / "carrier_compensation/checkpoints"
    latest_path = checkpoint_root / "LATEST.json"
    if latest_path.is_file():
        latest = json.loads(latest_path.read_text())
        if latest["candidate_id"] != candidate_id:
            raise JS2BError("carrier checkpoint belongs to another semantic seed")
        codes = np.load(Path(latest["codes"]["path"]), allow_pickle=False).astype(np.int32)
        current_errors = np.load(Path(latest["errors"]["path"]), allow_pickle=False)
        history = list(latest["history"])
        start_pass = int(latest["passes_completed"]) + 1
        converged = bool(latest["converged"])
    else:
        codes = context.codes.copy()
        current_errors, outputs, frame0 = pose_candidate(context, candidate_frame1, codes)
        initial_root = checkpoint_root / "pass_00"
        atomic_npy(initial_root / "codes_n600.int32.npy", codes)
        atomic_npy(initial_root / "pose_errors_n32.float64.npy", current_errors)
        atomic_npy(initial_root / "pose_outputs_n32.float32.npy", outputs)
        atomic_npy(initial_root / "frame0_n32.uint8.npy", frame0)
        history = [
            {
                "pass": 0,
                "accepted_moves": 0,
                "pose_mean": float(current_errors.mean()),
                "stratified_pose_mean": stratified_mean(current_errors, context.sample_weights),
                "pose_delta_vs_custody_base": stratified_mean(
                    current_errors - context.base_pose_errors, context.sample_weights
                ),
                "codes": file_record(initial_root / "codes_n600.int32.npy"),
                "errors": file_record(initial_root / "pose_errors_n32.float64.npy"),
                "outputs": file_record(initial_root / "pose_outputs_n32.float32.npy"),
                "frame0": file_record(initial_root / "frame0_n32.uint8.npy"),
            }
        ]
        latest = {
            "schema": "ddm_js2b_carrier_checkpoint.v1",
            "candidate_id": candidate_id,
            "passes_completed": 0,
            "converged": False,
            "codes": history[-1]["codes"],
            "errors": history[-1]["errors"],
            "history": history,
        }
        atomic_json(latest_path, latest)
        start_pass = 1
        converged = False
    eligible = context.selector_choices[context.sample] == 0
    for pass_index in range(start_pass, MAX_COMPENSATION_PASSES + 1):
        if converged:
            break
        neighbour_errors = np.full((SAMPLE_N, len(MOVES)), np.inf, dtype=np.float64)
        for move_index, (dimension, delta) in enumerate(MOVES):
            moved = codes.copy()
            moved[context.sample[eligible], dimension] += delta
            valid = (moved[context.sample, dimension] >= -2048) & (moved[context.sample, dimension] <= 2047) & eligible
            if not np.any(valid):
                continue
            errors, _, _ = pose_candidate(context, candidate_frame1, np.clip(moved, -2048, 2047))
            neighbour_errors[valid, move_index] = errors[valid]
        winners = neighbour_errors.argmin(axis=1)
        winning_errors = neighbour_errors[np.arange(SAMPLE_N), winners]
        accepted = eligible & (winning_errors < current_errors)
        for local_index in np.flatnonzero(accepted):
            dimension, delta = MOVES[int(winners[local_index])]
            codes[int(context.sample[local_index]), dimension] += delta
        checked_errors, outputs, frame0 = pose_candidate(context, candidate_frame1, codes)
        if np.any(checked_errors > current_errors + 1e-12):
            raise JS2BError("mixed carrier acceptance failed exact pose recheck")
        current_errors = checked_errors
        pass_root = checkpoint_root / f"pass_{pass_index:02d}"
        atomic_npy(pass_root / "codes_n600.int32.npy", codes)
        atomic_npy(pass_root / "pose_errors_n32.float64.npy", current_errors)
        atomic_npy(pass_root / "pose_outputs_n32.float32.npy", outputs)
        atomic_npy(pass_root / "frame0_n32.uint8.npy", frame0)
        row = {
            "pass": pass_index,
            "accepted_moves": int(accepted.sum()),
            "pose_mean": float(current_errors.mean()),
            "stratified_pose_mean": stratified_mean(current_errors, context.sample_weights),
            "pose_delta_vs_custody_base": stratified_mean(
                current_errors - context.base_pose_errors, context.sample_weights
            ),
            "codes": file_record(pass_root / "codes_n600.int32.npy"),
            "errors": file_record(pass_root / "pose_errors_n32.float64.npy"),
            "outputs": file_record(pass_root / "pose_outputs_n32.float32.npy"),
            "frame0": file_record(pass_root / "frame0_n32.uint8.npy"),
        }
        history.append(row)
        converged = not bool(accepted.any())
        latest = {
            "schema": "ddm_js2b_carrier_checkpoint.v1",
            "candidate_id": candidate_id,
            "passes_completed": pass_index,
            "converged": converged,
            "codes": row["codes"],
            "errors": row["errors"],
            "history": history,
        }
        atomic_json(latest_path, latest)
        print(json.dumps({"stage": "carrier_compensation", **row, "converged": converged}, sort_keys=True), flush=True)
    result = {
        "schema": "ddm_js2b_carrier_compensation.v1",
        "candidate_id": candidate_id,
        "semantic_sha256": sha256_bytes(semantic_blob),
        "complete": converged,
        "converged": converged,
        "passes_completed": int(latest["passes_completed"]),
        "max_passes": MAX_COMPENSATION_PASSES,
        "history": history,
        "selected_codes": latest["codes"],
        "selected_errors": latest["errors"],
        "base_pose_mean": stratified_mean(context.base_pose_errors, context.sample_weights),
        "candidate_pose_mean": stratified_mean(current_errors, context.sample_weights),
        "pose_delta": stratified_mean(current_errors - context.base_pose_errors, context.sample_weights),
        "pose_guard": POSE_GUARD,
        "pose_guard_pass": stratified_mean(current_errors - context.base_pose_errors, context.sample_weights)
        < POSE_GUARD,
        "selector_rows_frozen": context.sample[~eligible].tolist(),
        "axis": AXIS,
        "score_claim": False,
    }
    atomic_json(result_path, result)
    return result


def build_archive(
    context: Context,
    output: Path,
    screen: dict[str, Any],
    compensation: dict[str, Any],
) -> dict[str, Any]:
    result_path = output / "candidate/BUILD_RESULT.json"
    if result_path.is_file():
        result = json.loads(result_path.read_text())
        if result.get("archive") != file_record(Path(result["archive"]["path"])):
            raise JS2BError("candidate build checkpoint lost its archive")
        return result
    if not compensation["converged"]:
        raise JS2BError("cannot build an optimal-form proposal before carrier convergence")
    candidate_id, records, semantic_blob = selected_records(context, screen)
    codes = np.load(Path(compensation["selected_codes"]["path"]), allow_pickle=False).astype(np.int32)
    candidate_payload = context.modules.baseline.BaselinePayload(
        archive_path=BASE_ARCHIVE,
        semantic_blob=context.modules.baseline.encode_legacy_w4(records),
        carrier_blob=context.parts.carrier_blob,
        hpac_blob=context.parts.hpac_blob,
        token_stream=context.parts.token_stream,
        records=records,
    )
    cpr1 = context.modules.candidate_cpr1(context.canonical_carrier, codes)
    cap1, cap1_report = context.modules.coefficient_codec.encode_cap1(cpr1, frames=N, dimensions=D)
    carrier = context.modules.carrier.pack_frame0_selector_carrier(cap1, context.selector)
    raw_models = context.modules.book_residual._models(
        candidate_payload,
        context.parts.hpac_blob,
        semantic_blob=semantic_blob,
        carrier_blob=carrier,
        fixed_wans_ar1_rc64_schema=True,
    )
    if not raw_models.startswith(b"F24S") or len(raw_models) < 4 + 16_593 + 36_040:
        raise JS2BError("candidate did not produce the fixed F24S physical model")
    hpac = raw_models[4 : 4 + 16_593]
    semantic = raw_models[4 + 16_593 : 4 + 16_593 + 36_040]
    carrier_physical = raw_models[4 + 16_593 + 36_040 :]
    packed_carrier, metadata_report = context.modules.cp135.pack_cap1_metadata(carrier_physical)
    model_payload, model_report = context.modules.cp135._optimal_split_models(
        (hpac, semantic, packed_carrier),
        variant=candidate_id,
        representation="packed_cap1_metadata",
        output=output / "candidate",
        brotli_binary=str(BROTLI),
    )
    residual = context.parts.residual_payload[4:]
    token = context.parts.token_stream
    member = model_payload + residual + token
    archive = context.modules.cp135.deterministic_zip(member)
    repeat = context.modules.cp135.deterministic_zip(member)
    root = output / "candidate/retained"
    payloads = {
        "semantic_wans1": (root / "semantic.wans1", semantic_blob),
        "carrier_cpr1": (root / "carrier.cpr1", cpr1),
        "carrier_cap1": (root / "carrier.cap1", cap1),
        "carrier_with_selector": (root / "carrier.f0c1", carrier),
        "models_f24s_raw": (root / "models.f24s.raw", raw_models),
        "models_split_brotli": (root / "models.split_brotli", model_payload),
        "residual": (root / "residual.compact.bin", residual),
        "token": (root / "tokens.rc64", token),
        "member": (root / "p", member),
        "archive": (root / "archive.zip", archive),
        "repeat_archive": (root / "archive.repeat.zip", repeat),
    }
    records_out = {}
    for label, (path, value) in payloads.items():
        atomic_bytes(path, value)
        records_out[label] = file_record(path)
    if archive != repeat or context.modules.cp135.read_stored_member(payloads["archive"][0]) != member:
        raise JS2BError("candidate archive repeat or ZIP parse-back failed")
    restored = context.modules.residual.read_residual_archive(payloads["archive"][0])
    restored_cap1, restored_selector = context.modules.carrier.split_frame0_selector_carrier(restored.carrier_blob)
    identities = {
        "semantic": restored.semantic_blob == semantic_blob,
        "carrier_cpr1": context.modules.coefficient_codec.decode_cap1(restored_cap1, frames=N, dimensions=D) == cpr1,
        "selector": restored_selector == context.selector,
        "hpac": restored.hpac_blob == context.parts.hpac_blob,
        "residual": restored.residual_payload == context.parts.residual_payload,
        "tokens": restored.token_stream == context.parts.token_stream,
    }
    if not all(identities.values()):
        raise JS2BError(f"candidate failed shipped CP135 receiver parse-back: {identities}")
    screen_row = next(row for row in screen["rows"] if row["candidate_id"] == candidate_id)
    final_argmax = np.load(Path(screen_row["argmax"]["path"]), allow_pickle=False)
    final_logits = np.load(Path(screen_row["logits"]["path"]), allow_pickle=False)
    atomic_npy(root / "argmax_n32.npy", final_argmax)
    atomic_npy(root / "logits_n32.float32.npy", final_logits)
    records_out["argmax"] = file_record(root / "argmax_n32.npy")
    records_out["logits"] = file_record(root / "logits_n32.float32.npy")
    result = {
        "schema": "ddm_js2b_candidate_build.v1",
        "candidate_id": candidate_id,
        "archive": records_out["archive"],
        "repeat_archive": records_out["repeat_archive"],
        "repeat_byte_identical": True,
        "archive_delta_bytes": len(archive) - BASE_ARCHIVE_BYTES,
        "payloads": records_out,
        "cap1_report": cap1_report,
        "packed_cap1_metadata_report": metadata_report,
        "model_report": model_report,
        "receiver_parseback": identities,
        "all_payloads_retained": True,
        "axis": AXIS,
        "score_claim": False,
    }
    atomic_json(result_path, result)
    return result


def finalize(
    context: Context,
    output: Path,
    calibration: dict[str, Any],
    screen: dict[str, Any],
    compensation: dict[str, Any],
    build: dict[str, Any],
) -> dict[str, Any]:
    candidate_id = str(screen["selected_for_carrier_compensation"])
    row = next(item for item in screen["rows"] if item["candidate_id"] == candidate_id)
    metrics = row["metrics"]
    delta_bytes = int(build["archive_delta_bytes"])
    projected_delta = int(metrics["projected_n600_delta_flips"])
    projected_robust_delta = int(metrics["projected_n600_robust_delta_flips"])
    delta_dseg = projected_robust_delta / (N * H * W)
    base_pose = float(compensation["base_pose_mean"])
    pose_delta = float(compensation["pose_delta"])
    candidate_pose = base_pose + pose_delta
    delta_s = (
        100.0 * delta_dseg
        + math.sqrt(10.0 * max(candidate_pose, 0.0))
        - math.sqrt(10.0 * max(base_pose, 0.0))
        + 25.0 * delta_bytes / RATE_DENOMINATOR
    )
    gate = fire_gate(projected_robust_delta, delta_bytes, pose_delta)
    byte_per_robust_flip = delta_bytes / -projected_robust_delta if projected_robust_delta < 0 else None
    aggregate_beneficial = sum(int(item["metrics"]["beneficial_flips"]) for item in screen["rows"])
    aggregate_fragile = sum(int(item["metrics"]["tie_fragile_beneficial_flips"]) for item in screen["rows"])
    aggregate_fragile_fraction = aggregate_fragile / aggregate_beneficial if aggregate_beneficial else None
    f2_fired = (byte_per_robust_flip is not None and byte_per_robust_flip > 3.0) or (
        projected_robust_delta >= 0 and delta_bytes > 0
    )
    ranked = []
    for item in sorted(
        screen["rows"],
        key=lambda value: (
            int(value["metrics"]["projected_n600_robust_delta_flips"]),
            int(value["metrics"]["projected_n600_delta_flips"]),
            value["candidate_id"],
        ),
    ):
        ranked.append(
            {
                "rank": len(ranked) + 1,
                "candidate_id": item["candidate_id"],
                "projected_n600_delta_flips": item["metrics"]["projected_n600_delta_flips"],
                "projected_n600_robust_delta_flips": item["metrics"]["projected_n600_robust_delta_flips"],
                "tie_fragile_fraction_of_beneficial": item["metrics"]["tie_fragile_fraction_of_beneficial"],
                "carrier_compensated": item["candidate_id"] == candidate_id,
                "archive_delta_bytes": delta_bytes if item["candidate_id"] == candidate_id else None,
                "pose_delta": pose_delta if item["candidate_id"] == candidate_id else None,
                "disposition": (
                    "QUEUED-WITH-A-FIRE-ORDER"
                    if item["candidate_id"] == candidate_id and gate
                    else "FOLDED_BELOW_T4_GATE"
                    if item["candidate_id"] == candidate_id
                    else "FOLDED_SEMANTIC_SCREEN_ONLY"
                ),
            }
        )
    if gate:
        follow_on = {
            "disposition": "QUEUED-WITH-A-FIRE-ORDER",
            "owner": "MAIN exact contest-row owner",
            "consumer_store": str(output / "t4_acceptance"),
            "fire_trigger": (
                "MAIN owns the sole T4 acceptance lane; candidate archive bytes and SHA match; "
                "then run one paired n600 contest-CUDA row against the pinned CP135 base"
            ),
            "archive": build["archive"],
            "producer_dispatched_modal": False,
        }
    else:
        follow_on = {
            "disposition": "FOLDED",
            "owner": "ddm_js2b",
            "consumer_store": str(output / "FINAL_RESULT.json"),
            "fire_trigger": "none; charter T4 gate did not pass",
            "producer_dispatched_modal": False,
        }
    result = {
        "schema": "ddm_js2b_final_result.v1",
        "axis": AXIS,
        "score_claim": False,
        "sample": context.sample.tolist(),
        "sample_weights": context.sample_weights.tolist(),
        "calibration": calibration,
        "ranked_proposals": ranked,
        "selected_candidate": candidate_id,
        "selected_metrics": {
            "projected_n600_delta_flips": projected_delta,
            "projected_n600_robust_delta_flips": projected_robust_delta,
            "projected_local_relative_delta_dseg_robust": delta_dseg,
            "archive_bytes": build["archive"]["bytes"],
            "archive_delta_bytes": delta_bytes,
            "pose_base_n32": base_pose,
            "pose_candidate_n32": candidate_pose,
            "pose_delta_n32": pose_delta,
            "stratified_n32_policy_transfer_estimate_delta_s": delta_s,
            "delta_s_estimate_promotable": False,
            "delta_s_estimate_limitation": (
                "carrier compensation was solved only on the sampled rows; the estimate assumes "
                "the same policy can be solved on all n600 rows"
            ),
            "bytes_per_projected_margin_robust_flip": byte_per_robust_flip,
            "margin_distribution": metrics["beneficial_margin_distribution"],
            "tie_fragile_fraction_of_beneficial": metrics["tie_fragile_fraction_of_beneficial"],
        },
        "t4_acceptance_gate": {
            "pass": gate,
            "required_projected_robust_delta_flips_lte": T4_FLIP_GATE,
            "required_archive_delta_bytes_lte": T4_BYTE_GATE,
            "required_pose_delta_lt": POSE_GUARD,
        },
        "follow_on": follow_on,
        "falsifiers": {
            "F1": {
                "fired": (aggregate_fragile_fraction is not None and aggregate_fragile_fraction >= 0.8),
                "scope": ("FORMULATION: fixed nine-seed two-FiLM catalog on the stratified-random n32 CP135 sample"),
                "beneficial_flip_denominator": aggregate_beneficial,
                "tie_fragile_beneficial_flips": aggregate_fragile,
                "fraction": aggregate_fragile_fraction,
            },
            "F2": {
                "fired": f2_fired,
                "scope": "INSTANCE: selected compensated candidate with exact CP135 coder",
                "bytes_per_projected_margin_robust_flip": byte_per_robust_flip,
                "price_wall": (
                    "positive 42-byte rate with zero projected margin-robust flips"
                    if projected_robust_delta >= 0 and delta_bytes > 0
                    else "more than 3 bytes per projected margin-robust flip"
                    if f2_fired
                    else None
                ),
            },
            "F3": {
                "fired": not bool(compensation["pose_guard_pass"]),
                "scope": "INSTANCE: selected compensated candidate on stratified-random n32 custody gauge",
                "all_stacks_tested": False,
                "family_verdict_allowed": False,
            },
        },
        "boundaries": {
            "absolute_local_dseg_progress_claim": False,
            "exact_cuda_score_measured": False,
            "modal_dispatched": False,
            "promoted_cuda_argmax_field_retained": False,
            "full_n600_local_scorer_owned": False,
            "verdict_scope": "stratified-random n32 proposal screen plus exact full-container byte price",
        },
        "receipts": {
            "calibration": file_record(output / "calibration/RESULT.json"),
            "semantic_screen": file_record(output / "semantic_screen/RESULT.json"),
            "carrier_compensation": file_record(output / "carrier_compensation/RESULT.json"),
            "candidate_build": file_record(output / "candidate/BUILD_RESULT.json"),
        },
    }
    atomic_json(output / "FINAL_RESULT.json", result)
    atomic_json(output / "T4_ACCEPTANCE_FIRE_ORDER.json", follow_on)
    return result


def update_state(output: Path, **updates: Any) -> None:
    path = output / "state.json"
    state = json.loads(path.read_text()) if path.is_file() else {}
    state.update(updates)
    state.update(
        {
            "schema": "ddm_js2b_state.v1",
            "arm": "ddm_js2b",
            "resumable": True,
            "base_archive_sha256": BASE_ARCHIVE_SHA256,
            "seed": SEED,
            "batch": BATCH,
            "threads": THREADS,
            "axis": AXIS,
            "score_claim": False,
        }
    )
    atomic_json(path, state)


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    update_state(output, status="CUSTODY_PREFLIGHT", started_unix_seconds=time.time())
    context = build_context(output)
    update_state(output, status="DELTA_CALIBRATION")
    calibration = calibrate_delta(context, output)
    update_state(output, status="SEMANTIC_SCREEN", delta=calibration["delta"])
    screen = screen_semantics(context, output, float(calibration["delta"]))
    update_state(
        output,
        status="CARRIER_COMPENSATION",
        selected_candidate=screen["selected_for_carrier_compensation"],
    )
    compensation = compensate_carrier(context, output, screen)
    update_state(output, status="CONTAINER_BUILD", compensation_converged=compensation["converged"])
    build = build_archive(context, output, screen, compensation)
    result = finalize(context, output, calibration, screen, compensation, build)
    update_state(
        output,
        status="COMPLETE",
        complete=True,
        archive=build["archive"],
        t4_disposition=result["follow_on"]["disposition"],
        finished_unix_seconds=time.time(),
    )
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    value.add_argument("--resume", action="store_true", help="verify and continue retained checkpoints")
    return value


def main() -> None:
    args = parser().parse_args()
    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
