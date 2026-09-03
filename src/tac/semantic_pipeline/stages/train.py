# SPDX-License-Identifier: MIT
"""Real scorer-aware torch training stage for bounded FPC2 runs."""

from __future__ import annotations

import dataclasses
import importlib
import importlib.util
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


def _atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    torch.save(value, temporary)
    if path.is_file() and file_fact(path)["sha256"] != file_fact(temporary)["sha256"]:
        temporary.unlink()
        raise PipelineBlocked(f"refusing to overwrite a different retained tensor payload: {path}")
    os.replace(temporary, path)


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
    spec = importlib.util.spec_from_file_location("fpc2_gt_frame_utils", UPSTREAM / "frame_utils.py")
    if spec is None or spec.loader is None:
        raise PipelineBlocked("cannot load upstream frame conversion")
    frame_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frame_utils)
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


def _state_sha(state: dict[str, torch.Tensor]) -> str:
    digest = __import__("hashlib").sha256()
    for name, value in state.items():
        digest.update(name.encode())
        digest.update(memoryview(value.detach().cpu().contiguous().numpy()).cast("B"))
    return digest.hexdigest()


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


def run_train_stage(request: TrainRequest) -> dict[str, Any]:
    """Run or resume a bounded scorer-aware stage and retain its EMA payload."""

    if request.pair_count < 1 or request.pair_count > 8:
        raise PipelineBlocked("local train stage is limited to the n<=8 smoke lane")
    if request.steps < 1:
        raise ValueError("training steps must be positive")
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
    decay = eval_ema_decay_run_geometry(
        {"mode": "decay_from_seed_fraction", "updates_per_run": request.steps, "target_seed_fraction": 0.1}
    )
    ema = EMA(model, decay=decay, warmup=False)
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
    _atomic_torch_save(ema_path, {"state_dict": ema_state, "sealed_contract": sealed_contract})
    archive_result = replace_semantic_state(
        request.source_archive,
        output / "retained" / "archive.zip",
        ema_state,
    )
    quantized_state = archive_result.pop("quantized_state")
    quantized_path = output / "retained" / "semantic_quantized_state.pt"
    _atomic_torch_save(
        quantized_path,
        {"state_dict": dict(quantized_state), "sealed_contract": sealed_contract},
    )
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
        "ema": {"implementation": "tac.training.EMA", "law": sealed_contract["ema_law"], "decay": decay, "warmup": False, "updates": ema._num_updates},
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
