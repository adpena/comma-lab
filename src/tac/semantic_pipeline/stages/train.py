# SPDX-License-Identifier: MIT
"""Real scorer-aware torch training stages for bounded and population runs."""

from __future__ import annotations

import dataclasses
import gc
import hashlib
import importlib
import importlib.util
import json
import math
import os
import random
import struct
import sys
import time
from pathlib import Path
from typing import Any

import av
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from tac.canonical_equations.evaluators import eval_ema_decay_run_geometry
from tac.differentiable_eval_roundtrip import (
    CameraLiftKernel,
    EvalRoundTripOrdering,
    apply_eval_roundtrip_during_training,
    patch_upstream_yuv6_globally,
)
from tac.subset_selection import (
    DEFAULT_STRATIFIED_BLOCKS,
    MODE_SEEDED_RANDOM,
    MODE_STRATIFIED,
    select,
)
from tac.training import EMA

from ..archive import replace_semantic_state
from ..contracts import PipelineBlocked, TargetLineage, atomic_json, file_fact, require_device

REPO = Path(__file__).resolve().parents[4]
UPSTREAM = REPO / "upstream"
SHIPPED = REPO / "submissions" / "semantic_joint_ctxmix"
TOKEN_FIELD = Path(
    "/Volumes/VertigoDataTier/pact/ddm_g8s_single_run_reproof/"
    "store_v2/retained/inputs/tokens.u8"
)
DALI_CACHE = Path(
    "/Volumes/VertigoDataTier/pact/ddm_pr130_train_20260809/"
    "caches/gt_cache_600_official_ada.pt"
)
DALI_CACHE_SHA256 = "382d7dfe38b37c0cc5017e5645032faa045af6924db66e0b67549cc96c840195"


@dataclasses.dataclass(frozen=True)
class TrainRequest:
    video: Path
    source_archive: Path
    output_dir: Path
    device: str
    pair_count: int
    steps: int
    seed: int
    lineage: TargetLineage
    resume: bool
    chunk_pairs: int | None = None
    selection_mode: str = MODE_STRATIFIED
    stratified_blocks: int = DEFAULT_STRATIFIED_BLOCKS
    verdict_batch: int = 32
    resume_from: Path | None = None
    stop_after_chunks: int | None = None


@dataclasses.dataclass(frozen=True)
class EMALawSeal:
    """Typed executable/sealed form of ``ema_decay_run_geometry_v1``."""

    name: str
    mode: str
    total_updates: int
    target_seed_fraction: float
    decay: float
    terminal_seed_coefficient: float
    warmup: bool

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    torch.save(value, temporary)
    if path.is_file() and file_fact(path)["sha256"] != file_fact(temporary)["sha256"]:
        temporary.unlink()
        raise PipelineBlocked(f"refusing to overwrite a different retained tensor payload: {path}")
    os.replace(temporary, path)


def _ema_law(total_updates: int) -> EMALawSeal:
    target_seed_fraction = 0.1
    decay = eval_ema_decay_run_geometry(
        {
            "mode": "decay_from_seed_fraction",
            "updates_per_run": total_updates,
            "target_seed_fraction": target_seed_fraction,
        }
    )
    terminal = decay**total_updates
    seal = EMALawSeal(
        name="ema_decay_run_geometry_v1",
        mode="constant_decay_from_seed_fraction",
        total_updates=total_updates,
        target_seed_fraction=target_seed_fraction,
        decay=decay,
        terminal_seed_coefficient=terminal,
        warmup=False,
    )
    if not math.isclose(terminal, target_seed_fraction, rel_tol=1e-12, abs_tol=0.0):
        raise PipelineBlocked("EMA law evaluator does not reproduce its sealed seed coefficient")
    return seal


def _construct_ema(model: torch.nn.Module, seal: EMALawSeal) -> EMA:
    """Construct EMA from the typed law, never from a constructor literal."""

    ema = EMA(model, decay=seal.decay, warmup=seal.warmup)
    if ema.warmup != seal.warmup or ema.decay != seal.decay:
        raise PipelineBlocked("executable EMA law differs from the sealed law")
    return ema


def build_chunk_schedule(
    pair_count: int,
    chunk_pairs: int,
    *,
    seed: int,
    mode: str,
    block_count: int = DEFAULT_STRATIFIED_BLOCKS,
) -> tuple[tuple[int, ...], ...]:
    """Build one full-population epoch from explicit selector calls.

    ``select`` deliberately returns indices in canonical order. Selecting each
    chunk from the shrinking remaining population keeps every chunk explicitly
    random/stratified while making the flattened schedule a true permutation.
    """

    if pair_count < 1:
        raise ValueError("pair_count must be positive")
    if chunk_pairs < 1 or chunk_pairs > 120:
        raise PipelineBlocked("training chunk size must be in [1, 120]")
    if mode not in {MODE_SEEDED_RANDOM, MODE_STRATIFIED}:
        raise PipelineBlocked("population training requires seeded_random or stratified_blocks")
    remaining = list(range(pair_count))
    chunks: list[tuple[int, ...]] = []
    ordinal = 0
    while remaining:
        take = min(chunk_pairs, len(remaining))
        selection = select(
            take,
            len(remaining),
            mode=mode,
            seed=seed + ordinal,
            block_count=min(block_count, len(remaining)),
        )
        chosen_positions = tuple(selection.indices)
        chunk = tuple(remaining[position] for position in chosen_positions)
        chunks.append(chunk)
        chosen = set(chosen_positions)
        remaining = [value for position, value in enumerate(remaining) if position not in chosen]
        ordinal += 1
    flattened = tuple(pair for chunk in chunks for pair in chunk)
    if sorted(flattened) != list(range(pair_count)):
        raise PipelineBlocked("chunk selector did not produce a full-population permutation")
    return tuple(chunks)


def _schedule_for_update(request: TrainRequest, update: int) -> tuple[int, int, tuple[int, ...]]:
    if request.chunk_pairs is None:
        raise AssertionError("chunk schedule requested for bounded trainer")
    chunks_per_epoch = math.ceil(request.pair_count / request.chunk_pairs)
    epoch = update // chunks_per_epoch
    chunk_ordinal = update % chunks_per_epoch
    chunks = build_chunk_schedule(
        request.pair_count,
        request.chunk_pairs,
        seed=request.seed + epoch * 1_000_003,
        mode=request.selection_mode,
        block_count=request.stratified_blocks,
    )
    return epoch, chunk_ordinal, chunks[chunk_ordinal]


def _load_frame_utils():
    spec = importlib.util.spec_from_file_location("fpc3_gt_frame_utils", UPSTREAM / "frame_utils.py")
    if spec is None or spec.loader is None:
        raise PipelineBlocked("cannot load upstream frame conversion")
    frame_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frame_utils)
    return frame_utils


def _materialize_av_pair_cache(video: Path, pair_count: int, output: Path) -> dict[str, Any]:
    """Retain a crash-resumable RGB24 pair cache decoded by the contest converter."""

    cache_dir = output / "retained" / "input_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"av_pairs_n{pair_count}.rgb24"
    partial = cache_path.with_suffix(cache_path.suffix + ".partial")
    progress_path = cache_dir / f"av_pairs_n{pair_count}.progress.json"
    manifest_path = cache_dir / f"av_pairs_n{pair_count}.manifest.json"
    video_fact = file_fact(video)
    expected_bytes = pair_count * 2 * 874 * 1164 * 3
    contract = {
        "schema": "ddm_fpc3_av_pair_cache.v1",
        "video_sha256": video_fact["sha256"],
        "pair_count": pair_count,
        "shape": [pair_count, 2, 874, 1164, 3],
        "dtype": "uint8",
        "decoder": "upstream.frame_utils.yuv420_to_rgb",
    }
    if cache_path.is_file() and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("contract") != contract or cache_path.stat().st_size != expected_bytes:
            raise PipelineBlocked("retained AV pair cache differs from this run contract")
        if manifest.get("cache") != file_fact(cache_path):
            raise PipelineBlocked("retained AV pair cache no longer matches its manifest")
        return manifest
    if cache_path.exists() or manifest_path.exists():
        raise PipelineBlocked("AV pair cache is only partially finalized")
    completed_pairs = 0
    if progress_path.is_file():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("contract") != contract or not partial.is_file():
            raise PipelineBlocked("AV cache resume cursor differs from this run contract")
        completed_pairs = int(progress["completed_pairs"])
        if not 0 <= completed_pairs <= pair_count:
            raise PipelineBlocked("AV cache resume cursor is outside the requested population")
    elif partial.exists():
        raise PipelineBlocked("AV cache partial exists without a resume cursor")
    mode = "r+" if partial.exists() else "w+"
    cache = np.memmap(
        partial,
        mode=mode,
        dtype=np.uint8,
        shape=(pair_count, 2, 874, 1164, 3),
    )
    frame_utils = _load_frame_utils()
    written_frames = completed_pairs * 2
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        for frame_index, frame in enumerate(container.decode(stream)):
            if frame_index < written_frames:
                continue
            if frame_index >= pair_count * 2:
                break
            rgb = frame_utils.yuv420_to_rgb(frame)
            cache[frame_index // 2, frame_index % 2] = np.asarray(rgb, dtype=np.uint8)
            written_frames = frame_index + 1
            if written_frames % 2 == 0:
                cache.flush()
                atomic_json(
                    progress_path,
                    {"contract": contract, "completed_pairs": written_frames // 2},
                )
    cache.flush()
    del cache
    if written_frames != pair_count * 2:
        raise PipelineBlocked("video decode ended before the requested AV cache population")
    os.replace(partial, cache_path)
    manifest = {
        "schema": "ddm_fpc3_av_pair_cache_manifest.v1",
        "contract": contract,
        "cache": file_fact(cache_path),
        "progress": {"completed_pairs": pair_count, "complete": True},
    }
    atomic_json(progress_path, {"contract": contract, "completed_pairs": pair_count, "complete": True})
    atomic_json(manifest_path, manifest)
    return manifest


def _materialize_semantic_target_cache(
    *,
    request: TrainRequest,
    output: Path,
    av_cache: np.memmap,
    dali: dict[str, torch.Tensor],
    segnet: torch.nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    """Retain semantic targets once so training chunks never re-score GT."""

    cache_dir = output / "retained" / "input_cache"
    cache_path = cache_dir / f"semantic_{request.lineage.semantic}_n{request.pair_count}.u8"
    partial = cache_path.with_suffix(cache_path.suffix + ".partial")
    progress_path = cache_path.with_suffix(cache_path.suffix + ".progress.json")
    manifest_path = cache_path.with_suffix(cache_path.suffix + ".manifest.json")
    contract = {
        "schema": "ddm_fpc3_semantic_target_cache.v1",
        "lineage": request.lineage.semantic,
        "pair_count": request.pair_count,
        "shape": [request.pair_count, 384, 512],
        "dtype": "uint8",
        "materialize_batch": request.verdict_batch,
        "av_decoder": "upstream.frame_utils.yuv420_to_rgb",
        "scorer": "frozen upstream SegNet",
    }
    expected_bytes = request.pair_count * 384 * 512
    if cache_path.is_file() and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("contract") != contract or cache_path.stat().st_size != expected_bytes:
            raise PipelineBlocked("retained semantic target cache differs from this run contract")
        if manifest.get("cache") != file_fact(cache_path):
            raise PipelineBlocked("retained semantic target cache no longer matches its manifest")
        return manifest
    if cache_path.exists() or manifest_path.exists():
        raise PipelineBlocked("semantic target cache is only partially finalized")
    completed_pairs = 0
    if progress_path.is_file():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("contract") != contract or not partial.is_file():
            raise PipelineBlocked("semantic target resume cursor differs from this run contract")
        completed_pairs = int(progress["completed_pairs"])
        if not 0 <= completed_pairs <= request.pair_count:
            raise PipelineBlocked("semantic target resume cursor is outside the requested population")
    elif partial.exists():
        raise PipelineBlocked("semantic target partial exists without a resume cursor")
    targets = np.memmap(
        partial,
        mode="r+" if partial.exists() else "w+",
        dtype=np.uint8,
        shape=(request.pair_count, 384, 512),
    )
    with torch.inference_mode():
        for start in range(completed_pairs, request.pair_count, request.verdict_batch):
            stop = min(request.pair_count, start + request.verdict_batch)
            if request.lineage.semantic == "av":
                gt = (
                    torch.from_numpy(np.asarray(av_cache[start:stop]).copy())
                    .permute(0, 1, 4, 2, 3)
                    .float()
                    .to(device)
                )
                batch = segnet(segnet.preprocess_input(gt)).argmax(dim=1)
            else:
                batch = dali["seg"][start:stop].to(device)
            targets[start:stop] = batch.to(torch.uint8).cpu().numpy()
            targets.flush()
            atomic_json(progress_path, {"contract": contract, "completed_pairs": stop})
    targets.flush()
    del targets
    os.replace(partial, cache_path)
    manifest = {
        "schema": "ddm_fpc3_semantic_target_cache_manifest.v1",
        "contract": contract,
        "cache": file_fact(cache_path),
        "progress": {"completed_pairs": request.pair_count, "complete": True},
    }
    atomic_json(
        progress_path,
        {"contract": contract, "completed_pairs": request.pair_count, "complete": True},
    )
    atomic_json(manifest_path, manifest)
    return manifest


def _device_rng_state(device: torch.device) -> torch.Tensor | None:
    if device.type == "cuda":
        return torch.cuda.get_rng_state(device).cpu()
    if device.type == "mps" and hasattr(torch.mps, "get_rng_state"):
        return torch.mps.get_rng_state().cpu()
    return None


def _restore_device_rng_state(device: torch.device, state: torch.Tensor | None) -> None:
    if state is None:
        return
    if device.type == "cuda":
        torch.cuda.set_rng_state(state, device)
    elif device.type == "mps" and hasattr(torch.mps, "set_rng_state"):
        torch.mps.set_rng_state(state)


def _decode_av_pairs(video: Path, pair_count: int) -> torch.Tensor:
    frame_utils = _load_frame_utils()
    frames: list[torch.Tensor] = []
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        for index, frame in enumerate(container.decode(stream)):
            frames.append(frame_utils.yuv420_to_rgb(frame))
            if index + 1 == pair_count * 2:
                break
    if len(frames) != pair_count * 2:
        raise PipelineBlocked("video decode ended before the requested pair scope")
    return torch.stack(frames).reshape(pair_count, 2, 874, 1164, 3)


def _load_receiver_state(archive: Path):
    for name in list(sys.modules):
        if name == "runtime" or name.startswith("runtime.") or name == "_f26_renderer":
            sys.modules.pop(name, None)
    sys.path.insert(0, str(SHIPPED))
    try:
        f26 = importlib.import_module("runtime.f26_inflate")
        carrier = importlib.import_module("runtime.carrier_repack")
        renderer = f26._load_renderer(SHIPPED / "cpr1")
        parts = f26.read_residual_archive(archive)
        carrier_blob, _selector = carrier.split_frame0_selector_carrier(parts.carrier_blob)
        canonical = carrier.materialize_cpr1(carrier_blob, renderer)
        semantic_pose = struct.pack("<II", 40_252, len(canonical)) + bytes(40_252) + canonical
        _, basis, coefficients = renderer.unpack_semantic_pose(semantic_pose)
        model = renderer.SemanticTokenRenderer(96)
        state = renderer.unpack_variant_semantic_or_none(parts.semantic_blob, model.state_dict())
        if state is None:
            records = f26.decode_wans1(parts.semantic_blob)
            state = {
                row.schema.name: torch.from_numpy(np.ascontiguousarray(row.values, dtype=np.float32))
                for row in records
            }
        model.load_state_dict(state, strict=True)
        return renderer, model, basis, coefficients
    finally:
        sys.path.pop(0)


def _load_scorers(device: torch.device):
    # This call must precede importing upstream.modules; modules.py captures the
    # rgb_to_yuv6 symbol at import time.
    yuv_patch = patch_upstream_yuv6_globally()
    if str(UPSTREAM) not in sys.path:
        sys.path.insert(0, str(UPSTREAM))
    modules = importlib.import_module("modules")
    segnet = modules.SegNet().eval().to(device)
    posenet = modules.PoseNet().eval().to(device)
    segnet.load_state_dict(load_file(modules.segnet_sd_path, device=str(device)))
    posenet.load_state_dict(load_file(modules.posenet_sd_path, device=str(device)))
    for parameter in list(segnet.parameters()) + list(posenet.parameters()):
        parameter.requires_grad_(False)
    return modules, segnet, posenet, yuv_patch


def _render_carriers(renderer, basis: torch.Tensor, coefficients: torch.Tensor, pair_count: int) -> torch.Tensor:
    normalized = renderer.normalized_basis(basis)
    carrier = torch.einsum("bk,kchw->bchw", coefficients[:pair_count], normalized)  # SUBSET_SELECTION_OK:bounded plumbing smoke over the first pair_count pairs (receipt labels it a contiguous-prefix smoke, score_claim=false, never a verdict); population runs are chunked over all 600 pairs
    carrier = carrier / float(renderer.CARRIER_DIM) ** 0.5
    camera = F.interpolate(
        (127.5 + renderer.CARRIER_AMPLITUDE * carrier).clamp(0.0, 255.0).round(),
        size=(renderer.CAMERA_H, renderer.CAMERA_W),
        mode="bicubic",
        align_corners=False,
    ).clamp(0.0, 255.0).round()
    return F.interpolate(camera, size=(384, 512), mode="bilinear", align_corners=False)


def _render_selected_carriers(
    renderer,
    basis: torch.Tensor,
    coefficients: torch.Tensor,
    pair_ids: tuple[int, ...],
) -> torch.Tensor:
    normalized = renderer.normalized_basis(basis)
    selected = coefficients[list(pair_ids)]
    carrier = torch.einsum("bk,kchw->bchw", selected, normalized)
    carrier = carrier / float(renderer.CARRIER_DIM) ** 0.5
    camera = F.interpolate(
        (127.5 + renderer.CARRIER_AMPLITUDE * carrier).clamp(0.0, 255.0).round(),
        size=(renderer.CAMERA_H, renderer.CAMERA_W),
        mode="bicubic",
        align_corners=False,
    ).clamp(0.0, 255.0).round()
    return F.interpolate(camera, size=(384, 512), mode="bilinear", align_corners=False)


def _state_sha(state: dict[str, torch.Tensor]) -> str:
    digest = __import__("hashlib").sha256()
    for name, value in state.items():
        digest.update(name.encode())
        digest.update(memoryview(value.detach().cpu().contiguous().numpy()).cast("B"))
    return digest.hexdigest()


def _retain_state_payload(
    path: Path,
    state: dict[str, torch.Tensor],
    sealed_contract: dict[str, Any],
) -> None:
    """Write once or validate an identical state after endpoint resume."""

    if path.is_file():
        retained = torch.load(path, map_location="cpu", weights_only=False)
        retained_state = dict(retained.get("state_dict", {}))
        if (
            retained.get("sealed_contract") != sealed_contract
            or _state_sha(retained_state) != _state_sha(state)
        ):
            raise PipelineBlocked(f"retained state payload differs from resumed endpoint: {path}")
        return
    _atomic_torch_save(path, {"state_dict": state, "sealed_contract": sealed_contract})


def render_driver_prefix(
    *,
    source_archive: Path,
    quantized_state_path: Path,
    destination: Path,
    pair_count: int,
) -> dict[str, Any]:
    """Render the compiled driver state without parsing the fresh archive.

    This is intentionally independent of ``semantic_pipeline.receiver``: the
    semantic state comes from the retained compiler output, while the unchanged
    carrier/token/selector inputs come from their explicitly pinned sources.
    """

    if pair_count < 1 or pair_count > 8:
        raise PipelineBlocked("direct driver rendering is limited to the n<=8 smoke lane")
    renderer, model, basis, coefficients = _load_receiver_state(source_archive)
    state_payload = torch.load(quantized_state_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state_payload["state_dict"], strict=True)
    model.eval()
    normalized_basis = renderer.normalized_basis(basis)
    raw_tokens = np.memmap(TOKEN_FIELD, mode="r", dtype=np.uint8, shape=(600, 384, 512))

    f26 = importlib.import_module("runtime.f26_inflate")
    carrier_repack = importlib.import_module("runtime.carrier_repack")
    selector_module = importlib.import_module("runtime.frame0_selector")
    parts = f26.read_residual_archive(source_archive)
    if parts.compensation_blob is not None:
        raise PipelineBlocked("direct driver renderer has no compiled compensation consumer")
    _carrier_blob, selector_blob = carrier_repack.split_frame0_selector_carrier(parts.carrier_blob)
    selector_modes, selector_indices = selector_module.decode_selector(selector_blob)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.partial.{os.getpid()}")
    if destination.exists() or temporary.exists():
        raise PipelineBlocked(f"refusing to overwrite driver render: {destination}")
    output = np.memmap(
        temporary,
        mode="w+",
        dtype=np.uint8,
        shape=(pair_count * 2, renderer.CAMERA_H, renderer.CAMERA_W, 3),
    )
    with torch.inference_mode():
        for pair in range(pair_count):
            index = torch.tensor([pair])
            token = torch.from_numpy(np.asarray(raw_tokens[pair : pair + 1]).copy()).long()
            master = (
                F.interpolate(
                    model(token, index),
                    size=(renderer.CAMERA_H, renderer.CAMERA_W),
                    mode="bilinear",
                    align_corners=False,
                )
                .clamp(0.0, 255.0)
                .round()
            )
            carrier = torch.einsum(
                "bk,kchw->bchw", coefficients[pair : pair + 1], normalized_basis
            ) / float(renderer.CARRIER_DIM) ** 0.5
            slave = (
                F.interpolate(
                    (127.5 + renderer.CARRIER_AMPLITUDE * carrier)
                    .clamp(0.0, 255.0)
                    .round(),
                    size=(renderer.CAMERA_H, renderer.CAMERA_W),
                    mode="bicubic",
                    align_corners=False,
                )
                .clamp(0.0, 255.0)
                .round()
            )
            output[2 * pair] = slave.to(torch.uint8).permute(0, 2, 3, 1).numpy()[0]
            output[2 * pair + 1] = master.to(torch.uint8).permute(0, 2, 3, 1).numpy()[0]
    for mode_index, mode in enumerate(selector_modes):
        frame_ids = np.flatnonzero(selector_indices[:pair_count] == mode_index)  # SUBSET_SELECTION_OK:bounded plumbing smoke over the first pair_count pairs (receipt labels it a contiguous-prefix smoke, score_claim=false, never a verdict); population runs are chunked over all 600 pairs
        if frame_ids.size:
            output[2 * frame_ids] = selector_module.apply_pixel_mode(
                np.asarray(output[2 * frame_ids]).copy(), mode
            )
    output.flush()
    del output
    os.replace(temporary, destination)
    report = {
        "schema": "ddm_fpc2_direct_driver_render.v1",
        "status": "PASS",
        "score_claim": False,
        "pair_count": pair_count,
        "source_archive": file_fact(source_archive),
        "semantic_quantized_state": file_fact(quantized_state_path),
        "token_field": file_fact(TOKEN_FIELD),
        "raw": file_fact(destination),
    }
    atomic_json(destination.with_suffix(".driver.json"), report)
    return report


def _run_bounded_train_stage(request: TrainRequest) -> dict[str, Any]:
    """Run or resume a bounded scorer-aware stage and retain its EMA payload."""

    if request.pair_count < 1 or request.pair_count > 8:
        raise PipelineBlocked("local train stage is limited to the n<=8 smoke lane")
    if request.steps < 1:
        raise ValueError("training steps must be positive")
    if request.verdict_batch < 1 or request.verdict_batch > 120:
        raise PipelineBlocked("verdict batch must be in [1, 120]")
    request.lineage.__post_init__()
    binding = require_device(request.device)
    gradient_device = torch.device(binding.torch_device)
    random.seed(request.seed)
    np.random.seed(request.seed)
    torch.manual_seed(request.seed)
    torch.use_deterministic_algorithms(True)
    started = time.monotonic()
    output = request.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_fact = file_fact(request.source_archive)
    video_fact = file_fact(request.video)
    cache_fact = file_fact(DALI_CACHE)
    if cache_fact["sha256"] != DALI_CACHE_SHA256:
        raise PipelineBlocked("DALI target cache content no longer matches #1142 lineage pin")
    renderer, model, basis, coefficients = _load_receiver_state(request.source_archive)
    model = model.to(gradient_device).train()
    raw_tokens = np.memmap(TOKEN_FIELD, mode="r", dtype=np.uint8, shape=(600, 384, 512))
    tokens = torch.from_numpy(np.asarray(raw_tokens[: request.pair_count]).copy()).long().to(gradient_device)  # SUBSET_SELECTION_OK:bounded plumbing smoke over the first pair_count pairs (receipt labels it a contiguous-prefix smoke, score_claim=false, never a verdict); population runs are chunked over all 600 pairs
    pair_ids = torch.arange(request.pair_count, device=gradient_device)
    gt_av = _decode_av_pairs(request.video, request.pair_count)
    _, segnet, posenet, _patch = _load_scorers(gradient_device)
    gt_btchw = gt_av.permute(0, 1, 4, 2, 3).float().to(gradient_device)
    with torch.no_grad():
        semantic_target = segnet(segnet.preprocess_input(gt_btchw)).argmax(dim=1)
    if request.lineage.semantic == "dali":
        semantic_target = torch.load(DALI_CACHE, map_location=gradient_device, weights_only=True)["seg"][: request.pair_count].long()  # SUBSET_SELECTION_OK:bounded plumbing smoke over the first pair_count pairs (receipt labels it a contiguous-prefix smoke, score_claim=false, never a verdict); population runs are chunked over all 600 pairs
    dali = torch.load(DALI_CACHE, map_location=gradient_device, weights_only=True)
    pose_target = dali["pose"][: request.pair_count]  # SUBSET_SELECTION_OK:bounded plumbing smoke over the first pair_count pairs (receipt labels it a contiguous-prefix smoke, score_claim=false, never a verdict); population runs are chunked over all 600 pairs
    carrier_frames = _render_carriers(renderer, basis, coefficients, request.pair_count).to(gradient_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-7)
    ema_law = _ema_law(request.steps)
    decay = ema_law.decay
    ema = _construct_ema(model, ema_law)
    start_step = 0
    loss_rows: list[dict[str, float]] = []
    checkpoint_dir = output / "checkpoints"
    all_existing = sorted(checkpoint_dir.glob("stage_train_step_*.pt"))
    if all_existing and not request.resume:
        raise PipelineBlocked(
            "training checkpoints already exist; pass the matching resume contract instead of overwriting them"
        )
    existing = all_existing if request.resume else []
    if existing:
        checkpoint = torch.load(existing[-1], map_location=gradient_device, weights_only=False)
        sealed = checkpoint["sealed_contract"]
        expected = {
            "source_archive_sha256": source_fact["sha256"],
            "video_sha256": video_fact["sha256"],
            "pair_count": request.pair_count,
            "steps": request.steps,
            "seed": request.seed,
            "device": request.device,
            "target_lineage": request.lineage.as_dict(),
            "ema_law": "ema_decay_run_geometry_v1:decay_from_seed_fraction:0.1",
            "ema_decay": decay,
        }
        if sealed != expected:
            raise PipelineBlocked("training checkpoint contract differs from this run")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        ema.shadow = {name: value.to(gradient_device) for name, value in checkpoint["ema_shadow"].items()}
        ema._num_updates = int(checkpoint["ema_updates"])
        start_step = int(checkpoint["step"])
        loss_rows = list(checkpoint["loss_rows"])
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        rng_keys = {"python_rng_state", "numpy_rng_state", "device_rng_state"}
        if rng_keys.issubset(checkpoint):
            random.setstate(checkpoint["python_rng_state"])
            np.random.set_state(checkpoint["numpy_rng_state"])
            _restore_device_rng_state(gradient_device, checkpoint["device_rng_state"])
        elif start_step < request.steps:
            raise PipelineBlocked(
                "legacy checkpoint lacks complete RNG state and cannot resume more updates"
            )
    sealed_contract = {
        "source_archive_sha256": source_fact["sha256"],
        "video_sha256": video_fact["sha256"],
        "pair_count": request.pair_count,
        "steps": request.steps,
        "seed": request.seed,
        "device": request.device,
        "target_lineage": request.lineage.as_dict(),
        "ema_law": "ema_decay_run_geometry_v1:decay_from_seed_fraction:0.1",
        "ema_decay": decay,
    }
    for step in range(start_step, request.steps):
        optimizer.zero_grad(set_to_none=True)
        masters = model(tokens, pair_ids)
        masters_roundtrip = apply_eval_roundtrip_during_training(
            masters,
            ordering=EvalRoundTripOrdering.CAMERA_UINT8,
            lift_kernel=CameraLiftKernel.BILINEAR,
        )
        candidate = torch.stack((carrier_frames, masters_roundtrip), dim=1)
        seg_logits = segnet(segnet.preprocess_input(candidate))
        pose_vector = posenet(posenet.preprocess_input(candidate))["pose"][:, :6]
        seg_loss = F.cross_entropy(seg_logits, semantic_target)
        pose_loss = F.mse_loss(pose_vector, pose_target)
        loss = seg_loss + 0.001 * pose_loss
        loss.backward()
        optimizer.step()
        ema.update(model)
        loss_rows.append(
            {"step": float(step + 1), "seg_cross_entropy": float(seg_loss.detach()), "pose_mse": float(pose_loss.detach()), "loss": float(loss.detach())}
        )
        checkpoint_path = checkpoint_dir / f"stage_train_step_{step + 1:04d}.pt"
        if checkpoint_path.exists():
            raise PipelineBlocked(f"refusing to overwrite stage checkpoint: {checkpoint_path}")
        _atomic_torch_save(
            checkpoint_path,
            {
                "schema": "ddm_fpc2_train_checkpoint.v1",
                "step": step + 1,
                "model": {name: value.detach().cpu() for name, value in model.state_dict().items()},
                "ema_shadow": {name: value.detach().cpu() for name, value in ema.state_dict().items()},
                "ema_updates": ema._num_updates,
                "optimizer": optimizer.state_dict(),
                "torch_rng_state": torch.get_rng_state(),
                "python_rng_state": random.getstate(),
                "numpy_rng_state": np.random.get_state(),
                "device_rng_state": _device_rng_state(gradient_device),
                "loss_rows": loss_rows,
                "sealed_contract": sealed_contract,
            },
        )
    ema.apply(model)
    ema_state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    ema_path = output / "retained" / "semantic_ema_state.pt"
    _retain_state_payload(ema_path, ema_state, sealed_contract)
    archive_result = replace_semantic_state(
        request.source_archive,
        output / "retained" / "archive.zip",
        ema_state,
    )
    quantized_state = archive_result.pop("quantized_state")
    quantized_path = output / "retained" / "semantic_quantized_state.pt"
    _retain_state_payload(quantized_path, dict(quantized_state), sealed_contract)
    result = {
        "schema": "ddm_fpc2_train_stage.v1",
        "status": "PASS",
        "axis": "[macOS-CPU training smoke]" if request.device == "cpu" else f"[{request.device} gradient-only training smoke]",
        "score_claim": False,
        "elapsed_seconds": time.monotonic() - started,
        "source_archive": source_fact,
        "video": video_fact,
        "device_binding": binding.as_dict(),
        "target_lineage": request.lineage.as_dict(),
        "target_cache": cache_fact,
        "gt_decoder": "upstream.frame_utils.yuv420_to_rgb",
        "pair_count": request.pair_count,
        "steps": request.steps,
        "loss_rows": loss_rows,
        "eval_roundtrip": {"ordering": "camera_uint8", "lift_kernel": "bilinear", "inside_loss": True},
        "differentiable_yuv6_patched_before_scorer_construction": True,
        "ema": {
            "implementation": "tac.training.EMA",
            "law": sealed_contract["ema_law"],
            "law_provenance": ema_law.as_dict(),
            "decay": decay,
            "warmup": ema_law.warmup,
            "updates": ema._num_updates,
        },
        "ema_state": file_fact(ema_path),
        "ema_state_sha256": _state_sha(ema_state),
        "quantized_state_sha256": _state_sha(dict(quantized_state)),
        "quantized_state": file_fact(quantized_path),
        "archive": archive_result,
        "resume_from_step": start_step,
        "per_stage_checkpoints": [file_fact(path) for path in sorted(checkpoint_dir.glob("stage_train_step_*.pt"))],
    }
    atomic_json(output / "TRAIN_RESULT.json", result)
    return result


def _chunk_tensors(
    *,
    pair_ids: tuple[int, ...],
    semantic_cache: np.memmap,
    raw_tokens: np.memmap,
    dali: dict[str, torch.Tensor],
    renderer,
    basis: torch.Tensor,
    coefficients: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ids = list(pair_ids)
    token = torch.from_numpy(np.asarray(raw_tokens[ids]).copy()).long().to(device)
    index = torch.tensor(ids, dtype=torch.long, device=device)
    semantic_target = torch.from_numpy(np.asarray(semantic_cache[ids]).copy()).long().to(device)
    pose_target = dali["pose"][ids].to(device)
    carrier = _render_selected_carriers(renderer, basis, coefficients, pair_ids).to(device)
    return token, index, semantic_target, pose_target, carrier


def _checkpoint_payload(
    *,
    model: torch.nn.Module,
    ema: EMA,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    completed_updates: int,
    request: TrainRequest,
    loss_rows: list[dict[str, Any]],
    sealed_contract: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "ddm_fpc3_chunked_train_checkpoint.v1",
        "cursor": _cursor_for_completed(request, completed_updates),
        "model": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "ema_shadow": {name: value.detach().cpu() for name, value in ema.state_dict().items()},
        "ema_updates": ema._num_updates,
        "optimizer": optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "device_rng_state": _device_rng_state(device),
        "loss_rows": loss_rows,
        "sealed_contract": sealed_contract,
    }


def _cursor_for_completed(request: TrainRequest, completed_updates: int) -> dict[str, Any]:
    next_epoch, next_chunk_ordinal, next_pair_ids = (
        _schedule_for_update(request, completed_updates)
        if completed_updates < request.steps
        else (
            completed_updates // math.ceil(request.pair_count / int(request.chunk_pairs or 1)),
            0,
            (),
        )
    )
    return {
        "completed_updates": completed_updates,
        "next_epoch": next_epoch,
        "next_chunk_ordinal": next_chunk_ordinal,
        "next_pair_ids": list(next_pair_ids),
    }


def _restore_checkpoint(
    *,
    checkpoint_path: Path,
    model: torch.nn.Module,
    ema: EMA,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    sealed_contract: dict[str, Any],
) -> tuple[int, list[dict[str, Any]]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("schema") != "ddm_fpc3_chunked_train_checkpoint.v1":
        raise PipelineBlocked("resume checkpoint is not an FPC3 chunk checkpoint")
    if checkpoint.get("sealed_contract") != sealed_contract:
        raise PipelineBlocked("training checkpoint contract differs from this run")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    ema.shadow = {name: value.to(device) for name, value in checkpoint["ema_shadow"].items()}
    ema._num_updates = int(checkpoint["ema_updates"])
    torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
    random.setstate(checkpoint["python_rng_state"])
    np.random.set_state(checkpoint["numpy_rng_state"])
    _restore_device_rng_state(device, checkpoint["device_rng_state"])
    completed = int(checkpoint["cursor"]["completed_updates"])
    if completed < 0 or completed > int(sealed_contract["steps"]):
        raise PipelineBlocked("resume checkpoint cursor is outside the sealed schedule")
    restored_request = TrainRequest(
        video=Path(sealed_contract["video_path"]),
        source_archive=Path(sealed_contract["source_archive_path"]),
        output_dir=checkpoint_path.parent.parent,
        device=str(sealed_contract["device"]),
        pair_count=int(sealed_contract["pair_count"]),
        steps=int(sealed_contract["steps"]),
        seed=int(sealed_contract["seed"]),
        lineage=TargetLineage(**sealed_contract["target_lineage"]),
        resume=True,
        chunk_pairs=int(sealed_contract["chunk_pairs"]),
        selection_mode=str(sealed_contract["selection_mode"]),
        stratified_blocks=int(sealed_contract["stratified_blocks"]),
        verdict_batch=int(sealed_contract["verdict_batch"]),
    )
    expected_cursor = _cursor_for_completed(restored_request, completed)
    if checkpoint["cursor"] != expected_cursor:
        raise PipelineBlocked("resume checkpoint cursor does not match the sealed chunk schedule")
    return completed, list(checkpoint["loss_rows"])


def _retain_chunked_verdict(
    *,
    request: TrainRequest,
    output: Path,
    model: torch.nn.Module,
    renderer,
    basis: torch.Tensor,
    coefficients: torch.Tensor,
    segnet: torch.nn.Module,
    posenet: torch.nn.Module,
    device: torch.device,
    semantic_cache: np.memmap,
    raw_tokens: np.memmap,
    dali: dict[str, torch.Tensor],
) -> dict[str, Any]:
    if request.verdict_batch < 1 or request.verdict_batch > 120:
        raise PipelineBlocked("verdict batch must be in [1, 120]")
    if request.pair_count == 600 and request.verdict_batch >= request.pair_count:
        raise PipelineBlocked("n600 verdict must never forward all 600 pairs at once")
    verdict_root = output / "retained" / "verdict"
    verdict_root.mkdir(parents=True, exist_ok=True)
    result_path = verdict_root / "VERDICT_RESULT.json"
    camera_path = verdict_root / "camera_eval_u8.raw"
    seg_path = verdict_root / "seg_argmax_u8.raw"
    pose_path = verdict_root / "pose6_f32.raw"
    final_paths = (camera_path, seg_path, pose_path)
    temporary_paths = [path.with_name(f".{path.name}.partial") for path in final_paths]
    progress_path = verdict_root / "VERDICT_PROGRESS.json"
    verdict_contract = {
        "schema": "ddm_fpc3_chunked_verdict_contract.v1",
        "pair_count": request.pair_count,
        "verdict_batch": request.verdict_batch,
        "target_lineage": request.lineage.as_dict(),
    }

    def build_result(
        mismatch_count: int,
        semantic_count: int,
        pose_error: float,
        pose_count: int,
    ) -> dict[str, Any]:
        return {
            "schema": "ddm_fpc3_chunked_verdict.v1",
            "contract": verdict_contract,
            "axis": (
                "[macOS-CPU exact-scorer bounded mechanism smoke; not a verdict]"
                if request.device == "cpu" and request.pair_count < 32
                else f"[{request.device} exact-scorer population candidate; promotion requires evaluator custody]"
            ),
            "score_claim": False,
            "pair_count": request.pair_count,
            "selection": (
                "contiguous prefix mechanism smoke; no population claim"
                if request.pair_count < 600
                else "full population"
            ),
            "verdict_batch": request.verdict_batch,
            "d_seg": mismatch_count / semantic_count,
            "d_pose": pose_error / pose_count,
            "camera": file_fact(camera_path),
            "seg_argmax": file_fact(seg_path),
            "pose6": file_fact(pose_path),
        }

    if result_path.is_file():
        retained = json.loads(result_path.read_text(encoding="utf-8"))
        if retained.get("contract") != verdict_contract:
            raise PipelineBlocked("retained verdict contract differs from this run")
        for key in ("camera", "seg_argmax", "pose6"):
            if retained.get(key) != file_fact(Path(retained[key]["path"])):
                raise PipelineBlocked("retained verdict payload no longer matches its receipt")
        return retained
    if any(path.exists() for path in final_paths):
        if not progress_path.is_file():
            raise PipelineBlocked("final verdict payload exists without a completion cursor")
        completion = json.loads(progress_path.read_text(encoding="utf-8"))
        if (
            completion.get("contract") != verdict_contract
            or int(completion.get("completed_pairs", -1)) != request.pair_count
        ):
            raise PipelineBlocked("partial verdict finalize state is not a complete matching run")
        for final, temporary in zip(final_paths, temporary_paths, strict=True):
            if final.is_file() and temporary.exists():
                raise PipelineBlocked("verdict has both final and partial copies of one payload")
            if not final.is_file() and temporary.is_file():
                os.replace(temporary, final)
            if not final.is_file():
                raise PipelineBlocked("verdict completion cursor is missing a payload")
        result = build_result(
            int(completion["mismatches"]),
            int(completion["semantic_elements"]),
            float(completion["pose_squared_error"]),
            int(completion["pose_elements"]),
        )
        atomic_json(result_path, result)
        return result
    completed_pairs = 0
    mismatches = 0
    semantic_elements = 0
    pose_squared_error = 0.0
    pose_elements = 0
    if progress_path.is_file():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("contract") != verdict_contract or not all(
            path.is_file() for path in temporary_paths
        ):
            raise PipelineBlocked("verdict resume cursor differs from this run contract")
        completed_pairs = int(progress["completed_pairs"])
        mismatches = int(progress["mismatches"])
        semantic_elements = int(progress["semantic_elements"])
        pose_squared_error = float(progress["pose_squared_error"])
        pose_elements = int(progress["pose_elements"])
        if not 0 <= completed_pairs <= request.pair_count:
            raise PipelineBlocked("verdict resume cursor is outside the requested population")
    elif any(path.exists() for path in temporary_paths):
        raise PipelineBlocked("verdict partial exists without a resume cursor")
    mmap_mode = "r+" if completed_pairs else "w+"
    camera_out = np.memmap(
        temporary_paths[0],
        mode=mmap_mode,
        dtype=np.uint8,
        shape=(request.pair_count, 2, 3, 384, 512),
    )
    seg_out = np.memmap(
        temporary_paths[1],
        mode=mmap_mode,
        dtype=np.uint8,
        shape=(request.pair_count, 384, 512),
    )
    pose_out = np.memmap(
        temporary_paths[2],
        mode=mmap_mode,
        dtype=np.float32,
        shape=(request.pair_count, 6),
    )
    model.eval()
    with torch.inference_mode():
        for start in range(completed_pairs, request.pair_count, request.verdict_batch):
            stop = min(request.pair_count, start + request.verdict_batch)
            pair_ids = tuple(range(start, stop))
            tokens, ids, semantic_target, pose_target, carrier = _chunk_tensors(
                pair_ids=pair_ids,
                semantic_cache=semantic_cache,
                raw_tokens=raw_tokens,
                dali=dali,
                renderer=renderer,
                basis=basis,
                coefficients=coefficients,
                device=device,
            )
            masters = apply_eval_roundtrip_during_training(
                model(tokens, ids),
                ordering=EvalRoundTripOrdering.CAMERA_UINT8,
                lift_kernel=CameraLiftKernel.BILINEAR,
            )
            candidate = torch.stack((carrier, masters), dim=1)
            seg_argmax = segnet(segnet.preprocess_input(candidate)).argmax(dim=1)
            pose = posenet(posenet.preprocess_input(candidate))["pose"][:, :6]
            mismatches += int((seg_argmax != semantic_target).sum().cpu())
            semantic_elements += semantic_target.numel()
            pose_squared_error += float(((pose - pose_target) ** 2).sum().cpu())
            pose_elements += pose_target.numel()
            camera_out[start:stop] = candidate.clamp(0.0, 255.0).round().to(torch.uint8).cpu().numpy()
            seg_out[start:stop] = seg_argmax.to(torch.uint8).cpu().numpy()
            pose_out[start:stop] = pose.float().cpu().numpy()
            for payload in (camera_out, seg_out, pose_out):
                payload.flush()
            atomic_json(
                progress_path,
                {
                    "contract": verdict_contract,
                    "completed_pairs": stop,
                    "mismatches": mismatches,
                    "semantic_elements": semantic_elements,
                    "pose_squared_error": pose_squared_error,
                    "pose_elements": pose_elements,
                },
            )
    for payload in (camera_out, seg_out, pose_out):
        payload.flush()
    del camera_out, seg_out, pose_out
    for temporary, final in zip(temporary_paths, final_paths, strict=True):
        os.replace(temporary, final)
    result = build_result(mismatches, semantic_elements, pose_squared_error, pose_elements)
    atomic_json(result_path, result)
    return result


def _run_chunked_train_stage(request: TrainRequest) -> dict[str, Any]:
    if request.chunk_pairs is None:
        raise AssertionError("chunked trainer requires chunk_pairs")
    if request.pair_count < 1 or request.pair_count > 600:
        raise PipelineBlocked("chunked trainer pair count must be in [1, 600]")
    if request.steps < 1:
        raise ValueError("training steps must be positive")
    if request.verdict_batch < 1 or request.verdict_batch > 120:
        raise PipelineBlocked("verdict batch must be in [1, 120]")
    if request.stop_after_chunks is not None and request.stop_after_chunks < 1:
        raise ValueError("stop_after_chunks must be positive")
    request.lineage.__post_init__()
    binding = require_device(request.device)
    device = torch.device(binding.torch_device)
    random.seed(request.seed)
    np.random.seed(request.seed)
    torch.manual_seed(request.seed)
    torch.use_deterministic_algorithms(True)
    started = time.monotonic()
    output = request.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_fact = file_fact(request.source_archive)
    video_fact = file_fact(request.video)
    cache_fact = file_fact(DALI_CACHE)
    if cache_fact["sha256"] != DALI_CACHE_SHA256:
        raise PipelineBlocked("DALI target cache content no longer matches #1142 lineage pin")
    av_cache_manifest = _materialize_av_pair_cache(request.video, request.pair_count, output)
    implementation_facts = {
        "trainer": file_fact(Path(__file__)),
        "archive_builder": file_fact(Path(__file__).parents[1] / "archive.py"),
        "eval_roundtrip": file_fact(REPO / "src" / "tac" / "differentiable_eval_roundtrip.py"),
        "upstream_modules": file_fact(UPSTREAM / "modules.py"),
        "upstream_frame_utils": file_fact(UPSTREAM / "frame_utils.py"),
        "segnet_weights": file_fact(UPSTREAM / "models" / "segnet.safetensors"),
        "posenet_weights": file_fact(UPSTREAM / "models" / "posenet.safetensors"),
        "receiver_loader": file_fact(SHIPPED / "runtime" / "f26_inflate.py"),
        "receiver_renderer": file_fact(SHIPPED / "cpr1" / "inflate.py"),
    }
    av_cache_path = Path(av_cache_manifest["cache"]["path"])
    av_cache = np.memmap(
        av_cache_path,
        mode="r",
        dtype=np.uint8,
        shape=(request.pair_count, 2, 874, 1164, 3),
    )
    renderer, model, basis, coefficients = _load_receiver_state(request.source_archive)
    model = model.to(device).train()
    basis = basis.to(device)
    coefficients = coefficients.to(device)
    raw_tokens = np.memmap(TOKEN_FIELD, mode="r", dtype=np.uint8, shape=(600, 384, 512))
    dali = torch.load(DALI_CACHE, map_location="cpu", weights_only=True, mmap=True)
    _, segnet, posenet, _patch = _load_scorers(device)
    semantic_cache_manifest = _materialize_semantic_target_cache(
        request=request,
        output=output,
        av_cache=av_cache,
        dali=dali,
        segnet=segnet,
        device=device,
    )
    semantic_cache = np.memmap(
        Path(semantic_cache_manifest["cache"]["path"]),
        mode="r",
        dtype=np.uint8,
        shape=(request.pair_count, 384, 512),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-7)
    ema_law = _ema_law(request.steps)
    ema = _construct_ema(model, ema_law)
    epoch_zero = build_chunk_schedule(
        request.pair_count,
        request.chunk_pairs,
        seed=request.seed,
        mode=request.selection_mode,
        block_count=request.stratified_blocks,
    )
    schedule_bytes = json.dumps(epoch_zero, separators=(",", ":")).encode()
    sealed_contract = {
        "schema": "ddm_fpc3_chunked_train_contract.v1",
        "source_archive_path": str(request.source_archive.resolve()),
        "source_archive_sha256": source_fact["sha256"],
        "video_path": str(request.video.resolve()),
        "video_sha256": video_fact["sha256"],
        "token_field_sha256": file_fact(TOKEN_FIELD)["sha256"],
        "target_cache_sha256": cache_fact["sha256"],
        "av_cache_sha256": av_cache_manifest["cache"]["sha256"],
        "semantic_target_cache_sha256": semantic_cache_manifest["cache"]["sha256"],
        "pair_count": request.pair_count,
        "steps": request.steps,
        "seed": request.seed,
        "device": request.device,
        "target_lineage": request.lineage.as_dict(),
        "chunk_pairs": request.chunk_pairs,
        "selection_mode": request.selection_mode,
        "stratified_blocks": request.stratified_blocks,
        "epoch_zero_schedule_sha256": hashlib.sha256(schedule_bytes).hexdigest(),
        "verdict_batch": request.verdict_batch,
        "optimizer": {"name": "torch.optim.Adam", "lr": 1e-7},
        "loss": {"seg": "cross_entropy", "pose": "mse", "pose_weight": 0.001},
        "eval_roundtrip": {"ordering": "camera_uint8", "lift_kernel": "bilinear"},
        "ema_law": ema_law.as_dict(),
        "implementation_facts": implementation_facts,
    }
    checkpoint_dir = output / "checkpoints"
    existing = sorted(checkpoint_dir.glob("stage_train_chunk_*.pt"))
    resume_path = request.resume_from
    if resume_path is None and request.resume and existing:
        resume_path = existing[-1]
    if resume_path is not None and not resume_path.is_file():
        raise PipelineBlocked(f"resume checkpoint does not exist: {resume_path}")
    if existing and resume_path is None:
        raise PipelineBlocked("chunk checkpoints already exist; pass --resume-from instead of overwriting")
    start_update = 0
    loss_rows: list[dict[str, Any]] = []
    if resume_path is not None:
        if resume_path.resolve().parent != checkpoint_dir.resolve():
            raise PipelineBlocked("resume checkpoint must belong to this trainer output directory")
        start_update, loss_rows = _restore_checkpoint(
            checkpoint_path=resume_path,
            model=model,
            ema=ema,
            optimizer=optimizer,
            device=device,
            sealed_contract=sealed_contract,
        )
    for chunks_this_invocation, update in enumerate(range(start_update, request.steps), start=1):
        epoch, chunk_ordinal, pair_ids = _schedule_for_update(request, update)
        tokens, ids, semantic_target, pose_target, carrier = _chunk_tensors(
            pair_ids=pair_ids,
            semantic_cache=semantic_cache,
            raw_tokens=raw_tokens,
            dali=dali,
            renderer=renderer,
            basis=basis,
            coefficients=coefficients,
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        masters = model(tokens, ids)
        masters_roundtrip = apply_eval_roundtrip_during_training(
            masters,
            ordering=EvalRoundTripOrdering.CAMERA_UINT8,
            lift_kernel=CameraLiftKernel.BILINEAR,
        )
        candidate = torch.stack((carrier, masters_roundtrip), dim=1)
        seg_logits = segnet(segnet.preprocess_input(candidate))
        pose_vector = posenet(posenet.preprocess_input(candidate))["pose"][:, :6]
        seg_loss = F.cross_entropy(seg_logits, semantic_target)
        pose_loss = F.mse_loss(pose_vector, pose_target)
        loss = seg_loss + 0.001 * pose_loss
        loss.backward()
        optimizer.step()
        ema.update(model)
        loss_rows.append(
            {
                "update": update + 1,
                "epoch": epoch,
                "chunk_ordinal": chunk_ordinal,
                "pair_ids": list(pair_ids),
                "selection_mode": request.selection_mode,
                "seg_cross_entropy": float(seg_loss.detach()),
                "pose_mse": float(pose_loss.detach()),
                "loss": float(loss.detach()),
            }
        )
        checkpoint_path = checkpoint_dir / f"stage_train_chunk_{update + 1:06d}.pt"
        if checkpoint_path.exists():
            raise PipelineBlocked(f"refusing to overwrite stage checkpoint: {checkpoint_path}")
        _atomic_torch_save(
            checkpoint_path,
            _checkpoint_payload(
                model=model,
                ema=ema,
                optimizer=optimizer,
                device=device,
                completed_updates=update + 1,
                request=request,
                loss_rows=loss_rows,
                sealed_contract=sealed_contract,
            ),
        )
        if (
            request.stop_after_chunks is not None
            and chunks_this_invocation >= request.stop_after_chunks
            and update + 1 < request.steps
        ):
            interrupted = {
                "schema": "ddm_fpc3_chunked_train_interruption.v1",
                "status": "INTERRUPTED_AT_REQUESTED_BOUNDARY",
                "score_claim": False,
                "completed_updates": update + 1,
                "checkpoint": file_fact(checkpoint_path),
                "live_state_sha256": _state_sha(dict(model.state_dict())),
                "ema_state_sha256": _state_sha(ema.state_dict()),
                "sealed_contract": sealed_contract,
                "all_payloads_retained": True,
            }
            atomic_json(output / f"INTERRUPTED_AT_{update + 1:06d}.json", interrupted)
            return interrupted
    live_state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    live_state_sha256 = _state_sha(live_state)
    ema.apply(model)
    ema_state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    ema_path = output / "retained" / "semantic_ema_state.pt"
    _retain_state_payload(ema_path, ema_state, sealed_contract)
    archive_result = replace_semantic_state(
        request.source_archive,
        output / "retained" / "archive.zip",
        ema_state,
    )
    quantized_state = archive_result.pop("quantized_state")
    quantized_path = output / "retained" / "semantic_quantized_state.pt"
    _retain_state_payload(quantized_path, dict(quantized_state), sealed_contract)
    verdict = _retain_chunked_verdict(
        request=request,
        output=output,
        model=model,
        renderer=renderer,
        basis=basis,
        coefficients=coefficients,
        segnet=segnet,
        posenet=posenet,
        device=device,
        semantic_cache=semantic_cache,
        raw_tokens=raw_tokens,
        dali=dali,
    )
    result = {
        "schema": "ddm_fpc3_chunked_train_stage.v1",
        "status": "PASS",
        "axis": verdict["axis"],
        "score_claim": False,
        "elapsed_seconds": time.monotonic() - started,
        "source_archive": source_fact,
        "video": video_fact,
        "device_binding": binding.as_dict(),
        "target_lineage": request.lineage.as_dict(),
        "target_cache": cache_fact,
        "implementation_facts": implementation_facts,
        "av_pair_cache": av_cache_manifest,
        "semantic_target_cache": semantic_cache_manifest,
        "gt_decoder": "upstream.frame_utils.yuv420_to_rgb",
        "pair_count": request.pair_count,
        "steps": request.steps,
        "chunk_pairs": request.chunk_pairs,
        "selection_mode": request.selection_mode,
        "epoch_zero_chunks": [list(chunk) for chunk in epoch_zero],
        "loss_rows": loss_rows,
        "eval_roundtrip": {"ordering": "camera_uint8", "lift_kernel": "bilinear", "inside_loss": True},
        "differentiable_yuv6_patched_before_scorer_construction": True,
        "ema": {
            "implementation": "tac.training.EMA",
            "law": ema_law.as_dict(),
            "updates": ema._num_updates,
            "sealed_matches_executable": True,
        },
        "live_state_sha256": live_state_sha256,
        "ema_state": file_fact(ema_path),
        "ema_state_sha256": _state_sha(ema_state),
        "quantized_state_sha256": _state_sha(dict(quantized_state)),
        "quantized_state": file_fact(quantized_path),
        "archive": archive_result,
        "verdict": verdict,
        "resume_from_update": start_update,
        "completed_updates": request.steps,
        "checkpoints": [file_fact(path) for path in sorted(checkpoint_dir.glob("stage_train_chunk_*.pt"))],
        "all_payloads_retained": True,
    }
    atomic_json(output / "TRAIN_RESULT.json", result)
    del segnet, posenet, model, optimizer, ema
    gc.collect()
    return result


def run_train_stage(request: TrainRequest) -> dict[str, Any]:
    """Run the bounded smoke or the explicit chunked population trainer."""

    if request.chunk_pairs is None:
        return _run_bounded_train_stage(request)
    return _run_chunked_train_stage(request)
