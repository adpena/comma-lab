#!/usr/bin/env python3
"""Retained, resumable T4 trainer for the EC2 oriented latent adapter.

This worker is deliberately a real contest-component trainer.  It restores the
semantic renderer from the exact CP135 archive, injects EC1's counted oriented
conditioner before the four nonlinear TokenBlocks, applies the public camera
lift and uint8 cliff, and differentiates through the frozen upstream SegNet.
Every tensor materialized by a training or endpoint pass is persisted on the
mounted retention volume before it can be discarded.

The worker does not claim an exact score.  Its full-n600 endpoint is a retained
``[contest-CUDA T4 frozen-SegNet] COMPONENT-ONLY`` measurement.  MAIN must still
package the selected counted payload and run the canonical EC1/evaluator chain.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import shutil
import sys
import time
import zipfile
from pathlib import Path
from types import ModuleType
from typing import Any, Final

import numpy as np
import torch
from torch import nn
from torch.nn import functional

from tac.differentiable_eval_roundtrip import (
    CameraLiftKernel,
    apply_camera_uint8_lift_during_training,
)
from tac.training import EMA

try:
    from experiments import ddm_ec1_implicit_edge_conditioning as ec1_design
    from experiments.ddm_ec1_runtime import ec1_latent_conditioner as ec1_runtime
except ModuleNotFoundError:
    import ddm_ec1_implicit_edge_conditioning as ec1_design  # type: ignore[no-redef]
    from ddm_ec1_runtime import ec1_latent_conditioner as ec1_runtime  # type: ignore[no-redef]


REMOTE_REPO: Final = Path("/workspace/pact")
UPSTREAM: Final = REMOTE_REPO / "upstream"
AXIS: Final = "[contest-CUDA T4 frozen-SegNet, n600] COMPONENT-ONLY"
SEED: Final = 20_260_814
N_PAIRS: Final = 600
CLASSES: Final = 5
WORK_HW: Final = (384, 512)
CAMERA_HW: Final = (874, 1164)
BASE_FLIPS: Final = 34_970
PIXELS: Final = N_PAIRS * WORK_HW[0] * WORK_HW[1]
ERROR_BALANCE_WEIGHT: Final = PIXELS / BASE_FLIPS
BASE_ARCHIVE_BYTES: Final = 186_252
BASE_ARCHIVE_SHA256: Final = "6eb1a3b79cb167e03372339e07e93cae13b6ba3114a9eb917288bb038622edb6"
TOKENS_BYTES: Final = 117_964_928
TOKENS_SHA256: Final = "03f5379d70e4bbd88e125cfbfb785cf5473315c70a5b78661fa426bb3e96e0f4"
GT_FIELD_BYTES: Final = 117_964_928
GT_FIELD_SHA256: Final = "91d3ff11a904c476b56a8be8af2225fb4a390d02fac9d3b09ef4704ad6e77248"
BASE_FIELD_BYTES: Final = 117_964_928
BASE_FIELD_SHA256: Final = "7648ad42e9f21942f86e81b97cabf46b710af747bba0909f7837ef3891232727"
EC1_ARCHIVE_EFFICIENCY_FLIPS_PER_BYTE: Final = 0.785
TRAIN_BATCH_SIZE: Final = 1
ENDPOINT_BATCH_SIZE: Final = 16
TOTAL_TRAIN_STEPS: Final = 1_800
EMA_DECAY: Final = EMA.decay_from_total_steps(TOTAL_TRAIN_STEPS)
MAX_WALL_SECONDS: Final = 10_500.0
EXPECTED_RETAINED_BYTES: Final = 40 * 1024**3
STORAGE_RESERVE_BYTES: Final = 4 * 1024**3

STAGES: Final = (
    {
        "name": "10_target_birth",
        "steps": 600,
        "learning_rate": 1.0e-3,
        "error_weight": ERROR_BALANCE_WEIGHT,
        "correct_weight": 0.25,
        "margin_weight": 0.0,
    },
    {
        "name": "20_balanced_descent",
        "steps": 600,
        "learning_rate": 3.0e-4,
        "error_weight": ERROR_BALANCE_WEIGHT,
        "correct_weight": 1.0,
        "margin_weight": 0.0,
    },
    {
        "name": "30_collateral_finish",
        "steps": 600,
        "learning_rate": 1.0e-4,
        "error_weight": 0.25 * ERROR_BALANCE_WEIGHT,
        "correct_weight": 1.0,
        "margin_weight": 0.0,
    },
)


class EC2WorkerError(RuntimeError):
    """An exact-receiver, trainer, retention, or resume invariant failed."""


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(path.name + f".partial.{os.getpid()}")
    with staging.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(staging, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_bytes(path, canonical_json_bytes(value))


def retain_exact_bytes(path: Path, payload: bytes) -> dict[str, Any]:
    if path.is_file():
        if path.read_bytes() != payload:
            raise EC2WorkerError(f"retained payload differs: {path}")
    else:
        atomic_bytes(path, payload)
    return file_record(path)


def atomic_npy(path: Path, value: Any) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(path.name + f".partial.{os.getpid()}")
    with staging.open("wb") as stream:
        np.save(stream, np.asarray(value), allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(staging, path)
    return file_record(path)


def retain_exact_npy(path: Path, value: Any) -> dict[str, Any]:
    expected = np.asarray(value)
    if path.is_file():
        retained = np.load(path, allow_pickle=False)
        if retained.dtype != expected.dtype or retained.shape != expected.shape:
            raise EC2WorkerError(f"retained NPY geometry differs: {path}")
        if not np.array_equal(retained, expected, equal_nan=True):
            raise EC2WorkerError(f"retained NPY values differ: {path}")
        return file_record(path)
    return atomic_npy(path, expected)


def atomic_torch_save(path: Path, value: Any) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(path.name + f".partial.{os.getpid()}")
    torch.save(value, staging)
    with staging.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(staging, path)
    return file_record(path)


def require_exact(path: Path, *, size: int, digest: str, label: str) -> None:
    if not path.is_file():
        raise EC2WorkerError(f"missing {label}: {path}")
    if path.stat().st_size != size or sha256_file(path) != digest:
        raise EC2WorkerError(f"{label} differs from pinned custody: {path}")


def _safe_extract(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            relative = Path(info.filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise EC2WorkerError(f"unsafe runtime ZIP member: {info.filename}")
            if info.is_dir():
                continue
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            payload = archive.read(info)
            if target.is_file():
                if target.read_bytes() != payload:
                    raise EC2WorkerError(f"runtime resume bytes differ: {target}")
                continue
            atomic_bytes(target, payload)
            os.chmod(target, (info.external_attr >> 16) & 0o777 or 0o644)


def configure_reproducibility() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise EC2WorkerError("EC2 authority-bound worker requires a CUDA T4; no fallback")
    if torch.cuda.get_device_name(0) != "Tesla T4":
        raise EC2WorkerError(f"EC2 requires Tesla T4; got {torch.cuda.get_device_name(0)!r}")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    precision_api = "legacy"
    try:
        torch.backends.cuda.matmul.fp32_precision = "ieee"
        torch.backends.cudnn.conv.fp32_precision = "ieee"
        precision_api = "fp32_precision"
        matmul_precision = torch.backends.cuda.matmul.fp32_precision
        cudnn_precision = torch.backends.cudnn.conv.fp32_precision
    except (AttributeError, RuntimeError):
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        matmul_precision = "ieee"
        cudnn_precision = "ieee"
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.use_deterministic_algorithms(False)
    return {
        "seed": SEED,
        "device": torch.cuda.get_device_name(0),
        "precision_api": precision_api,
        "cuda_matmul_fp32_precision": matmul_precision,
        "cudnn_conv_fp32_precision": cudnn_precision,
        "cudnn_benchmark": False,
        "cudnn_deterministic": False,
        "deterministic_algorithms": False,
        "reason": (
            "matches the pinned CP135 CUDA receiver rail; exact endpoint fields and "
            "archive repeats are the verdict, with all RNG seeded"
        ),
    }


def storage_preflight(run_root: Path) -> dict[str, Any]:
    retained = sum(path.stat().st_size for path in run_root.rglob("*") if path.is_file())
    required = max(0, EXPECTED_RETAINED_BYTES - retained) + STORAGE_RESERVE_BYTES
    usage = shutil.disk_usage(run_root)
    report = {
        "schema": "ddm_ec2_storage_preflight.v1",
        "tier": str(run_root),
        "free_bytes": usage.free,
        "already_retained_bytes": retained,
        "expected_total_retained_bytes": EXPECTED_RETAINED_BYTES,
        "reserve_bytes": STORAGE_RESERVE_BYTES,
        "required_free_bytes": required,
        "passed": usage.free >= required,
        "cleanup_policy": "certify-or-block; no training, scorer, camera, or archive payload is deleted",
    }
    atomic_json(run_root / "STORAGE_PREFLIGHT.json", report)
    if not report["passed"]:
        raise EC2WorkerError(f"storage preflight failed: free={usage.free}, required={required}")
    return report


def _load_renderer_module(renderer_dir: Path) -> ModuleType:
    path = renderer_dir / "inflate.py"
    name = "_ddm_ec2_cp135_renderer"
    cached = sys.modules.get(name)
    if cached is not None:
        return cached
    sys.path.insert(0, str(renderer_dir))
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise EC2WorkerError(f"cannot load exact CP135 renderer: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def load_exact_semantic(runtime_root: Path, archive_path: Path, device: torch.device) -> nn.Module:
    """Restore the exact semantic object consumed by CP135's public receiver."""
    sys.path.insert(0, str(runtime_root))
    try:
        from runtime.entropy.renderer_weight_codec import decode_wans1
        from runtime.residual_archive import read_residual_archive
    finally:
        sys.path.pop(0)
    renderer = _load_renderer_module(runtime_root / "cpr1")
    parts = read_residual_archive(archive_path)
    semantic = renderer.SemanticTokenRenderer(96)
    records = decode_wans1(parts.semantic_blob)
    state = {
        record.schema.name: torch.from_numpy(np.ascontiguousarray(record.values, dtype=np.float32))
        for record in records
    }
    semantic.load_state_dict(state, strict=True)
    semantic.eval().to(device)
    for parameter in semantic.parameters():
        parameter.requires_grad_(False)
    return semantic


def load_segnet(device: torch.device) -> nn.Module:
    from safetensors.torch import load_file

    sys.path.insert(0, str(UPSTREAM))
    try:
        from modules import SegNet, segnet_sd_path
    finally:
        sys.path.pop(0)
    scorer = SegNet().eval().to(device)
    scorer.load_state_dict(load_file(segnet_sd_path, device=str(device)))
    for parameter in scorer.parameters():
        parameter.requires_grad_(False)
    return scorer


def conditioned_semantic_forward(
    semantic: nn.Module,
    conditioner: ec1_runtime.LatentEdgeConditioner,
    tokens: torch.Tensor,
    pair_indices: torch.Tensor,
) -> torch.Tensor:
    """Inject the QAT adapter at EC1's exact pre-TokenBlock receiver site."""
    value = semantic.token_embed(tokens).permute(0, 3, 1, 2)
    value = semantic.coord_mix(
        torch.cat(
            [value, semantic.coordinates(value.shape[0], value.device, value.dtype)],
            dim=1,
        )
    )
    value = value + quantized_conditioner_forward(conditioner, ec1_runtime.edge_context(tokens, conditioner.family))
    frame = semantic.frame_embed(pair_indices)
    for block in semantic.blocks:
        value = block(value, frame)
    return torch.sigmoid(semantic.head(functional.gelu(value))) * 255.0


def _int8_weight_ste(value: torch.Tensor) -> torch.Tensor:
    """Forward-exact EC1 int8 tensor storage with an identity STE backward."""
    maximum = float(value.detach().abs().amax().cpu())
    scale_value = max(maximum / 127.0, 1.0e-8)
    scale = torch.tensor(scale_value, dtype=value.dtype, device=value.device)
    decoded = torch.clamp(torch.round(value / scale), -127, 127) * scale
    return value + (decoded - value).detach()


def _float16_ste(value: torch.Tensor | None) -> torch.Tensor | None:
    if value is None:
        return None
    decoded = value.to(torch.float16).to(value.dtype)
    return value + (decoded - value).detach()


def quantized_conditioner_forward(
    conditioner: ec1_runtime.LatentEdgeConditioner, context: torch.Tensor
) -> torch.Tensor:
    """Run the adapter through the exact EC1 serializer quantizers in-loop."""
    value = functional.conv2d(
        context,
        _int8_weight_ste(conditioner.context.weight),
        _float16_ste(conditioner.context.bias),
        padding=1,
    )
    value = functional.gelu(value)
    value = functional.conv2d(
        value,
        _int8_weight_ste(conditioner.depthwise.weight),
        _float16_ste(conditioner.depthwise.bias),
        padding=1,
        groups=conditioner.hidden,
    )
    value = functional.gelu(value)
    value = functional.conv2d(
        value,
        _int8_weight_ste(conditioner.head.weight),
        _float16_ste(conditioner.head.bias),
    )
    return torch.tanh(value) * conditioner.max_delta


def composite_receiver(
    semantic: nn.Module,
    conditioner: ec1_runtime.LatentEdgeConditioner,
    tokens: torch.Tensor,
    pair_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return pre-R, camera integer-valued float, and exact scorer-grid tensors."""
    pre_r = conditioned_semantic_forward(semantic, conditioner, tokens, pair_indices)
    camera = apply_camera_uint8_lift_during_training(
        pre_r,
        lift_kernel=CameraLiftKernel.BILINEAR,
        simulate_uint8=True,
        ste_round=True,
    )
    scorer_input = functional.interpolate(camera, size=WORK_HW, mode="bilinear", align_corners=False)
    return pre_r, camera, scorer_input


def initialize_conditioner(device: torch.device, family: str = "oriented") -> ec1_runtime.LatentEdgeConditioner:
    if family not in ec1_runtime.FAMILIES:
        raise EC2WorkerError(f"unsupported EC2 adapter family: {family!r}")
    torch.manual_seed(SEED + ec1_runtime.FAMILIES.index(family))
    model = ec1_runtime.LatentEdgeConditioner(hidden=4, max_delta=0.25, family=family).to(device)
    nn.init.kaiming_uniform_(model.context.weight, a=math.sqrt(5))
    nn.init.zeros_(model.context.bias)
    nn.init.dirac_(model.depthwise.weight)
    nn.init.zeros_(model.depthwise.bias)
    nn.init.zeros_(model.head.weight)
    if model.head.bias is not None:
        nn.init.zeros_(model.head.bias)
    return model.train()


def stratified_pair_order(base_field: np.ndarray, gt_field: np.ndarray, stage: int) -> list[int]:
    """Deterministically visit all 600 pairs once, interleaved across 20 hardness strata."""
    if base_field.shape != gt_field.shape or base_field.shape[0] != N_PAIRS:
        raise EC2WorkerError("base/GT argmax field geometry differs")
    counts = np.count_nonzero(base_field != gt_field, axis=(1, 2))
    ranked = np.argsort(counts, kind="stable")
    strata = [ranked[index::20].tolist() for index in range(20)]
    rng = np.random.default_rng(SEED + stage)
    for values in strata:
        rng.shuffle(values)
    result: list[int] = []
    for offset in range(max(map(len, strata))):
        for values in strata:
            if offset < len(values):
                result.append(int(values[offset]))
    if len(result) != N_PAIRS or sorted(result) != list(range(N_PAIRS)):
        raise EC2WorkerError("stratified order is not a full n600 permutation")
    return result


def realized_flip_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    base_error: torch.Tensor,
    *,
    error_weight: float,
    correct_weight: float,
    margin_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Differentiable target loss with extra pressure on CP135's realized errors."""
    pixel_ce = functional.cross_entropy(logits, target, reduction="none")
    weights = torch.where(
        base_error,
        torch.full_like(pixel_ce, error_weight),
        torch.full_like(pixel_ce, correct_weight),
    )
    loss = (weights * pixel_ce).mean()
    if margin_weight:
        selected = logits.gather(1, target[:, None]).squeeze(1)
        other = logits.masked_fill(
            functional.one_hot(target, CLASSES).permute(0, 3, 1, 2).bool(),
            float("-inf"),
        ).amax(dim=1)
        margin = functional.softplus(other - selected)
        loss = loss + margin_weight * (weights * margin).mean()
    predicted = logits.argmax(dim=1)
    hard_flips = int(torch.count_nonzero(predicted != target).item())
    fixed_base_errors = int(torch.count_nonzero(base_error & (predicted == target)).item())
    introduced_errors = int(torch.count_nonzero((~base_error) & (predicted != target)).item())
    return loss, {
        "loss": float(loss.detach().item()),
        "hard_flips": hard_flips,
        "fixed_base_errors": fixed_base_errors,
        "introduced_errors": introduced_errors,
    }


def retain_training_step(
    root: Path,
    *,
    pre_r: torch.Tensor,
    camera: torch.Tensor,
    scorer_input: torch.Tensor,
    logits: torch.Tensor,
    target: torch.Tensor,
    base_error: torch.Tensor,
    receipt: dict[str, Any],
) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    payloads = {
        "pre_r.npy": pre_r.detach().cpu().numpy().astype(np.float32, copy=False),
        "camera_uint8.npy": camera.detach().cpu().numpy().astype(np.uint8, copy=False),
        "scorer_input.npy": scorer_input.detach().cpu().numpy().astype(np.float32, copy=False),
        "logits.npy": logits.detach().cpu().numpy().astype(np.float32, copy=False),
        "argmax.npy": logits.detach().argmax(dim=1).cpu().numpy().astype(np.uint8, copy=False),
        "target.npy": target.detach().cpu().numpy().astype(np.uint8, copy=False),
        "base_error.npy": base_error.detach().cpu().numpy().astype(bool, copy=False),
    }
    records: dict[str, Any] = {}
    for name, value in payloads.items():
        path = root / name
        records[name] = retain_exact_npy(path, value)
    result = {**receipt, "payloads": records}
    retain_exact_bytes(root / "STEP_RECEIPT.json", canonical_json_bytes(result))
    return result


def _checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    ema: EMA,
    *,
    global_step: int,
    stage_index: int,
    stage_step: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = {
        "schema": "ddm_ec2_training_config.v1",
        "family": model.family,
        "hidden": model.hidden,
        "max_delta": model.max_delta,
        "seed": SEED,
        "stages": list(STAGES),
        "total_steps": TOTAL_TRAIN_STEPS,
        "ema_decay": EMA_DECAY,
        "ema_warmup": True,
        "optimizer": "AdamW(weight_decay=0)",
        "train_batch_size": TRAIN_BATCH_SIZE,
        "endpoint_batch_size": ENDPOINT_BATCH_SIZE,
        "base_archive_sha256": BASE_ARCHIVE_SHA256,
        "tokens_sha256": TOKENS_SHA256,
        "gt_field_sha256": GT_FIELD_SHA256,
        "base_field_sha256": BASE_FIELD_SHA256,
        "receiver": "CP135 semantic + EC1 pre-TokenBlock QAT adapter + BILINEAR camera uint8 R",
    }
    live = {
        "schema": "ddm_ec2_live_checkpoint.v1",
        "seed": SEED,
        "family": model.family,
        "global_step": global_step,
        "stage_index": stage_index,
        "stage_step": stage_step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
        "config": config,
    }
    shadow = {
        "schema": "ddm_ec2_ema_checkpoint.v1",
        "seed": SEED,
        "family": model.family,
        "global_step": global_step,
        "stage_index": stage_index,
        "stage_step": stage_step,
        "decay": ema.decay,
        "warmup": ema.warmup,
        "num_updates": ema._num_updates,
        "shadow": ema.state_dict(),
        "config": config,
    }
    return live, shadow


def save_checkpoint_pair(
    run_root: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    ema: EMA,
    *,
    global_step: int,
    stage_index: int,
    stage_step: int,
    label: str,
) -> dict[str, Any]:
    live, shadow = _checkpoint_payload(
        model,
        optimizer,
        ema,
        global_step=global_step,
        stage_index=stage_index,
        stage_step=stage_step,
    )
    root = run_root / "checkpoints" / label
    if root.exists():
        suffix = 1
        while (run_root / "checkpoints" / f"{label}.resume{suffix:03d}").exists():
            suffix += 1
        root = run_root / "checkpoints" / f"{label}.resume{suffix:03d}"
    live_record = atomic_torch_save(root / "live.pt", live)
    ema_record = atomic_torch_save(root / "ema.pt", shadow)
    pointer = {
        "schema": "ddm_ec2_checkpoint_pointer.v1",
        "global_step": global_step,
        "stage_index": stage_index,
        "stage_step": stage_step,
        "live": live_record,
        "ema": ema_record,
    }
    atomic_json(run_root / "checkpoints/LATEST.json", pointer)
    return pointer


def resume_checkpoint(
    run_root: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    ema: EMA,
    device: torch.device,
) -> dict[str, int]:
    pointer_path = run_root / "checkpoints/LATEST.json"
    if not pointer_path.is_file():
        return {"global_step": 0, "stage_index": 0, "stage_step": 0}
    pointer = json.loads(pointer_path.read_text())
    live = torch.load(pointer["live"]["path"], map_location=device, weights_only=False)
    shadow = torch.load(pointer["ema"]["path"], map_location=device, weights_only=False)
    expected_config = _checkpoint_payload(
        model,
        optimizer,
        ema,
        global_step=int(live["global_step"]),
        stage_index=int(live["stage_index"]),
        stage_step=int(live["stage_step"]),
    )[0]["config"]
    if live.get("config") != expected_config or shadow.get("config") != expected_config:
        raise EC2WorkerError("resume checkpoint training config differs")
    model.load_state_dict(live["model"], strict=True)
    optimizer.load_state_dict(live["optimizer"])
    torch.set_rng_state(live["torch_rng"])
    torch.cuda.set_rng_state_all(live["cuda_rng"])
    np.random.set_state(live["numpy_rng"])
    random.setstate(live["python_rng"])
    ema.shadow = {name: value.to(device) for name, value in shadow["shadow"].items()}
    ema._num_updates = int(shadow["num_updates"])
    return {
        "global_step": int(live["global_step"]),
        "stage_index": int(live["stage_index"]),
        "stage_step": int(live["stage_step"]),
    }


def snapshot_model(model: nn.Module, state: dict[str, torch.Tensor]) -> nn.Module:
    result = ec1_runtime.LatentEdgeConditioner(hidden=model.hidden, max_delta=model.max_delta, family=model.family)
    result.load_state_dict({name: value.detach().cpu() for name, value in state.items()})
    return result.eval()


def package_stage(
    run_root: Path,
    *,
    stage_name: str,
    base_archive: Path,
    model: nn.Module,
    ema: EMA,
) -> dict[str, Any]:
    stage_root = run_root / "stages" / stage_name
    results: dict[str, Any] = {}
    for kind, state in (("live", model.state_dict()), ("ema", ema.state_dict())):
        snapshot = snapshot_model(model, state)
        serialized = ec1_design.serialize_module(snapshot, stage_root, label=kind)
        coded = Path(serialized["coded"]["path"])
        retained = stage_root / "retained" / f"{kind}.ec1_latent.int8.br"
        retain_exact_bytes(retained, coded.read_bytes())
        archive = ec1_design.deterministic_archive(base_archive, retained.read_bytes())
        archive_path = stage_root / "retained" / f"{kind}.archive.zip"
        repeat_path = stage_root / "retained" / f"{kind}.archive.repeat.zip"
        retain_exact_bytes(archive_path, archive)
        retain_exact_bytes(
            repeat_path,
            ec1_design.deterministic_archive(base_archive, retained.read_bytes()),
        )
        if archive_path.read_bytes() != repeat_path.read_bytes():
            raise EC2WorkerError(f"stage archive repeat differs: {stage_name}/{kind}")
        results[kind] = {
            "serialized": serialized,
            "module": file_record(retained),
            "archive": file_record(archive_path),
            "archive_repeat": file_record(repeat_path),
            "archive_delta_bytes": archive_path.stat().st_size - BASE_ARCHIVE_BYTES,
        }
    atomic_json(stage_root / "STAGE_PACKAGE.json", results)
    return results


@torch.inference_mode()
def full_endpoint(
    run_root: Path,
    *,
    semantic: nn.Module,
    model: nn.Module,
    scorer: nn.Module,
    tokens: np.ndarray,
    target: np.ndarray,
    base_field: np.ndarray,
    device: torch.device,
) -> dict[str, Any]:
    root = run_root / "endpoint/retained"
    root.mkdir(parents=True, exist_ok=True)
    progress_path = run_root / "endpoint/PROGRESS.json"
    progress = json.loads(progress_path.read_text()) if progress_path.is_file() else {"completed_pairs": 0}
    completed_pairs = int(progress["completed_pairs"])
    model.eval().to(device)
    started = time.monotonic()
    batch_rows: list[dict[str, Any]] = []
    for start in range(0, N_PAIRS, ENDPOINT_BATCH_SIZE):
        end = min(N_PAIRS, start + ENDPOINT_BATCH_SIZE)
        batch_root = root / f"batches/batch_{start // ENDPOINT_BATCH_SIZE:04d}"
        receipt_path = batch_root / "BATCH_RECEIPT.json"
        if end <= completed_pairs:
            if not receipt_path.is_file():
                raise EC2WorkerError(f"endpoint progress lacks batch receipt: {receipt_path}")
            batch_rows.append(json.loads(receipt_path.read_text()))
            continue
        index = torch.arange(start, end, dtype=torch.long, device=device)
        token_batch = torch.from_numpy(np.asarray(tokens[start:end]).copy()).long().to(device)
        pre_r, camera, scorer_input = composite_receiver(semantic, model, token_batch, index)
        logits = scorer(scorer_input)
        payloads = {
            "pre_r.npy": pre_r.cpu().numpy().astype(np.float32, copy=False),
            "camera_uint8.npy": camera.cpu().numpy().astype(np.uint8, copy=False),
            "scorer_input.npy": scorer_input.cpu().numpy().astype(np.float32, copy=False),
            "logits.npy": logits.cpu().numpy().astype(np.float32, copy=False),
            "argmax.npy": logits.argmax(dim=1).cpu().numpy().astype(np.uint8, copy=False),
            "target.npy": np.asarray(target[start:end], dtype=np.uint8),
            "base_error.npy": np.asarray(base_field[start:end] != target[start:end]),
        }
        records = {name: retain_exact_npy(batch_root / name, value) for name, value in payloads.items()}
        row = {
            "schema": "ddm_ec2_endpoint_batch.v1",
            "pair_start": start,
            "pair_end": end,
            "batch_size": end - start,
            "payloads": records,
        }
        retain_exact_bytes(receipt_path, canonical_json_bytes(row))
        batch_rows.append(row)
        atomic_json(
            progress_path,
            {
                "schema": "ddm_ec2_endpoint_progress.v1",
                "completed_pairs": end,
                "endpoint_batch_size": ENDPOINT_BATCH_SIZE,
            },
        )
        completed_pairs = end
    argmax = np.concatenate(
        [np.load(Path(row["payloads"]["argmax.npy"]["path"]), allow_pickle=False) for row in batch_rows],
        axis=0,
    )
    if argmax.shape != (N_PAIRS, *WORK_HW):
        raise EC2WorkerError(f"endpoint n600 argmax geometry differs: {argmax.shape}")
    argmax_record = retain_exact_npy(root / "argmax_n600.npy", argmax)
    batch_manifest = retain_exact_bytes(
        root / "BATCH_RECEIPTS.jsonl",
        b"".join(canonical_json_bytes(row) for row in batch_rows),
    )
    errors = argmax != target
    base_errors = base_field != target
    flips = int(np.count_nonzero(errors))
    fixed = int(np.count_nonzero(base_errors & ~errors))
    introduced = int(np.count_nonzero(~base_errors & errors))
    result = {
        "schema": "ddm_ec2_full_endpoint.v1",
        "axis": AXIS,
        "n_pairs": N_PAIRS,
        "batch_size": ENDPOINT_BATCH_SIZE,
        "flips": flips,
        "base_instrument_flips": BASE_FLIPS,
        "net_flip_reduction_vs_instrument": BASE_FLIPS - flips,
        "fixed_base_errors": fixed,
        "introduced_errors": introduced,
        "elapsed_seconds_this_resume": time.monotonic() - started,
        "retained_payloads": {
            "argmax_n600": argmax_record,
            "batch_receipts": batch_manifest,
        },
        "all_pre_r_camera_scorer_logit_argmax_target_batches_retained": True,
        "score_claim": False,
        "promotion_eligible": False,
    }
    atomic_json(run_root / "endpoint/ENDPOINT_RESULT.json", result)
    return result


def copy_selected(run_root: Path, package: dict[str, Any], endpoint: dict[str, Any]) -> dict[str, Any]:
    root = run_root / "stages/selected/retained"
    root.mkdir(parents=True, exist_ok=True)
    sources = {
        "ec1_latent.int8.br": Path(package["ema"]["module"]["path"]),
        "archive.zip": Path(package["ema"]["archive"]["path"]),
        "archive.repeat.zip": Path(package["ema"]["archive_repeat"]["path"]),
    }
    records: dict[str, Any] = {}
    for name, source in sources.items():
        target = root / name
        records[name] = retain_exact_bytes(target, source.read_bytes())
    delta = records["archive.zip"]["bytes"] - BASE_ARCHIVE_BYTES
    required = int(np.ceil(max(0, delta) * EC1_ARCHIVE_EFFICIENCY_FLIPS_PER_BYTE))
    reduction = int(endpoint["net_flip_reduction_vs_instrument"])
    family = str(package["ema"]["serialized"]["family"])
    clears = reduction >= required
    result = {
        "schema": "ddm_ec2_selected_candidate.v1",
        "family": family,
        "payloads": records,
        "archive_delta_bytes": delta,
        "break_even_required_flip_reduction": required,
        "measured_flip_reduction": reduction,
        "clears_family_break_even": clears,
        "clears_oriented_break_even": family == "oriented" and clears,
        "controls_may_fire": family == "oriented" and clears,
        "score_claim": False,
        "next_consumer": "EC1 package command, then RE1T/JS1B exact component measurement",
    }
    atomic_json(run_root / "stages/selected/SELECTED_RESULT.json", result)
    return result


def train(run_root: Path) -> dict[str, Any]:
    started = time.monotonic()
    storage = storage_preflight(run_root)
    reproducibility = configure_reproducibility()
    device = torch.device("cuda:0")
    inputs = run_root / "inputs"
    request = json.loads((inputs / "REQUEST.json").read_text())
    family = str(request["family"])
    if family not in ec1_runtime.FAMILIES:
        raise EC2WorkerError(f"sealed request names unsupported family: {family!r}")
    archive = inputs / "cp135.archive.zip"
    runtime_bundle = inputs / "cp135.runtime.zip"
    tokens_path = inputs / "decoded_tokens_n600.npy"
    gt_field_path = inputs / "gt_argmax_n600.npy"
    base_field_path = inputs / "base_argmax_n600.npy"
    require_exact(archive, size=BASE_ARCHIVE_BYTES, digest=BASE_ARCHIVE_SHA256, label="CP135 archive")
    require_exact(tokens_path, size=TOKENS_BYTES, digest=TOKENS_SHA256, label="decoded tokens")
    require_exact(gt_field_path, size=GT_FIELD_BYTES, digest=GT_FIELD_SHA256, label="GT argmax field")
    require_exact(base_field_path, size=BASE_FIELD_BYTES, digest=BASE_FIELD_SHA256, label="base argmax field")
    runtime_root = run_root / "runtime/cp135"
    _safe_extract(runtime_bundle, runtime_root)
    semantic = load_exact_semantic(runtime_root, archive, device)
    scorer = load_segnet(device)
    tokens = np.load(tokens_path, mmap_mode="r", allow_pickle=False)
    target = np.load(gt_field_path, mmap_mode="r", allow_pickle=False)
    base_field = np.load(base_field_path, mmap_mode="r", allow_pickle=False)
    if tokens.shape != (N_PAIRS, *WORK_HW) or target.shape != tokens.shape or base_field.shape != tokens.shape:
        raise EC2WorkerError("n600 token/field geometry differs")

    model = initialize_conditioner(device, family)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(STAGES[0]["learning_rate"]), weight_decay=0.0)
    ema = EMA(model, decay=EMA_DECAY, warmup=True)
    state = resume_checkpoint(run_root, model, optimizer, ema, device)

    if state["global_step"] == 0:
        index = torch.tensor([stratified_pair_order(base_field, target, 0)[0]], device=device)
        token = torch.from_numpy(np.asarray(tokens[index.item()]).copy())[None].long().to(device)
        with torch.inference_mode():
            base_pre_r = semantic(token, index)
            adapted_pre_r = conditioned_semantic_forward(semantic, model, token, index)
        identity_root = run_root / "preflight/retained"
        identity = {
            "base_pre_r": retain_exact_npy(identity_root / "base_pre_r.npy", base_pre_r.cpu().numpy()),
            "adapted_pre_r": retain_exact_npy(identity_root / "adapted_pre_r.npy", adapted_pre_r.cpu().numpy()),
            "pair_index": int(index.item()),
            "exact_identity": bool(torch.equal(base_pre_r, adapted_pre_r)),
        }
        if not identity["exact_identity"]:
            raise EC2WorkerError(f"zero-initialized {family} adapter is not exact identity")
        atomic_json(run_root / "preflight/ZERO_IDENTITY.json", identity)

    final_package: dict[str, Any] | None = None
    for stage_index, stage in enumerate(STAGES):
        order = stratified_pair_order(base_field, target, stage_index)
        if stage_index < state["stage_index"]:
            continue
        start_step = state["stage_step"] if stage_index == state["stage_index"] else 0
        for group in optimizer.param_groups:
            group["lr"] = float(stage["learning_rate"])
        for stage_step in range(start_step, int(stage["steps"])):
            if time.monotonic() - started > MAX_WALL_SECONDS:
                save_checkpoint_pair(
                    run_root,
                    model,
                    optimizer,
                    ema,
                    global_step=state["global_step"],
                    stage_index=stage_index,
                    stage_step=stage_step,
                    label=f"paused_g{state['global_step']:04d}",
                )
                atomic_json(
                    run_root / "PAUSED.json",
                    {
                        "schema": "ddm_ec2_pause.v1",
                        "reason": "10,500-second worker budget reached before Modal's 3-hour hard cap",
                        "resume_same_run_id": True,
                        "global_step": state["global_step"],
                    },
                )
                raise EC2WorkerError("worker budget reached; resume same run_id from retained checkpoint")
            pair = order[stage_step]
            pair_index = torch.tensor([pair], dtype=torch.long, device=device)
            token = torch.from_numpy(np.asarray(tokens[pair]).copy())[None].long().to(device)
            target_tensor = torch.from_numpy(np.asarray(target[pair]).copy())[None].long().to(device)
            base_error = torch.from_numpy(np.asarray(base_field[pair] != target[pair]).copy())[None].to(device)
            optimizer.zero_grad(set_to_none=True)
            pre_r, camera, scorer_input = composite_receiver(semantic, model, token, pair_index)
            logits = scorer(scorer_input)
            loss, metrics = realized_flip_loss(
                logits,
                target_tensor,
                base_error,
                error_weight=float(stage["error_weight"]),
                correct_weight=float(stage["correct_weight"]),
                margin_weight=float(stage["margin_weight"]),
            )
            receipt = retain_training_step(
                run_root / f"training/retained/{stage['name']}/step_{stage_step:04d}_pair_{pair:03d}",
                pre_r=pre_r,
                camera=camera,
                scorer_input=scorer_input,
                logits=logits,
                target=target_tensor,
                base_error=base_error,
                receipt={
                    "schema": "ddm_ec2_training_step.v1",
                    "stage": stage,
                    "stage_index": stage_index,
                    "stage_step": stage_step,
                    "global_step_before_update": state["global_step"],
                    "pair_index": pair,
                    "metrics_before_update": metrics,
                },
            )
            loss.backward()
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item())
            optimizer.step()
            ema.update(model)
            state["global_step"] += 1
            state["stage_index"] = stage_index
            state["stage_step"] = stage_step + 1
            receipt["gradient_norm"] = gradient_norm
            retain_exact_bytes(
                run_root / f"training/receipts/global_{state['global_step']:04d}.json",
                canonical_json_bytes(receipt),
            )
            save_checkpoint_pair(
                run_root,
                model,
                optimizer,
                ema,
                global_step=state["global_step"],
                stage_index=stage_index,
                stage_step=stage_step + 1,
                label=f"periodic_g{state['global_step']:04d}",
            )
        final_package = package_stage(
            run_root, stage_name=str(stage["name"]), base_archive=archive, model=model, ema=ema
        )
        save_checkpoint_pair(
            run_root,
            model,
            optimizer,
            ema,
            global_step=state["global_step"],
            stage_index=stage_index + 1,
            stage_step=0,
            label=f"stage_{stage['name']}_complete",
        )
        state["stage_index"] = stage_index + 1
        state["stage_step"] = 0

    if final_package is None:
        package_path = run_root / f"stages/{STAGES[-1]['name']}/STAGE_PACKAGE.json"
        if not package_path.is_file():
            raise EC2WorkerError("completed resume is missing final stage package")
        final_package = json.loads(package_path.read_text())
    endpoint_module = Path(final_package["ema"]["module"]["path"])
    endpoint_model = ec1_runtime.load_conditioner(endpoint_module.read_bytes(), device)
    endpoint = full_endpoint(
        run_root,
        semantic=semantic,
        model=endpoint_model,
        scorer=scorer,
        tokens=tokens,
        target=target,
        base_field=base_field,
        device=device,
    )
    endpoint["counted_adapter_consumed"] = file_record(endpoint_module)
    endpoint["endpoint_uses_serialized_parseback_model"] = True
    atomic_json(run_root / "endpoint/ENDPOINT_RESULT.json", endpoint)
    selected = copy_selected(run_root, final_package, endpoint)
    final = {
        "schema": "ddm_ec2_adapter_trainer_result.v1",
        "axis": AXIS,
        "family": family,
        "storage_preflight": storage,
        "reproducibility": reproducibility,
        "schedule": list(STAGES),
        "schedule_derivation": {
            "measured_field_pass_anchor_seconds": 900,
            "pairs": 600,
            "derived_seconds_per_pair": 1.5,
            "conservative_training_multiplier_assumption": 3.0,
            "derived_training_seconds": 8_100,
            "endpoint_reserve_seconds": 900,
            "checkpoint_and_package_reserve_seconds": 900,
            "projected_seconds": 9_900,
            "hard_cap_seconds": 10_800,
            "estimated_t4_cost_usd_at_0_60_per_hour": 1.8,
        },
        "ema": {"decay": EMA_DECAY, "warmup": True, "updates": ema._num_updates},
        "endpoint": endpoint,
        "selected": selected,
        "elapsed_seconds": time.monotonic() - started,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
    }
    atomic_json(run_root / "FINAL_RESULT.json", final)
    return final


def run_toy_gate(output: Path) -> dict[str, Any]:
    """CPU structural gate only; never an empirical or scorer result."""
    torch.manual_seed(SEED)

    class TinyBlock(nn.Module):
        def forward(self, value: torch.Tensor, frame: torch.Tensor) -> torch.Tensor:
            return value + frame[:, : value.shape[1], None, None] * 0.0

    class TinySemantic(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.token_embed = nn.Embedding(CLASSES, 96)
            self.coord_mix = nn.Conv2d(100, 96, 1)
            self.frame_embed = nn.Embedding(3, 96)
            self.blocks = nn.ModuleList([TinyBlock() for _ in range(4)])
            self.head = nn.Conv2d(96, 3, 3, padding=1)

        @staticmethod
        def coordinates(batch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
            return torch.zeros(batch, 4, 8, 12, device=device, dtype=dtype)

        def forward(self, tokens: torch.Tensor, pair_indices: torch.Tensor) -> torch.Tensor:
            value = self.token_embed(tokens).permute(0, 3, 1, 2)
            value = self.coord_mix(torch.cat([value, self.coordinates(value.shape[0], value.device, value.dtype)], 1))
            frame = self.frame_embed(pair_indices)
            for block in self.blocks:
                value = block(value, frame)
            return torch.sigmoid(self.head(functional.gelu(value))) * 255.0

    semantic = TinySemantic().eval()
    for parameter in semantic.parameters():
        parameter.requires_grad_(False)
    adapter = initialize_conditioner(torch.device("cpu"))
    tokens = torch.randint(0, CLASSES, (1, 8, 12))
    index = torch.tensor([0])
    base = semantic(tokens, index)
    zero = conditioned_semantic_forward(semantic, adapter, tokens, index)
    if not torch.equal(base, zero):
        raise EC2WorkerError("toy zero identity failed")
    with torch.no_grad():
        adapter.head.weight[0, 0, 0, 0] = 0.5
    moved = conditioned_semantic_forward(semantic, adapter, tokens, index)
    if torch.equal(base, moved):
        raise EC2WorkerError("toy nonzero conditioner did not move the semantic receiver")
    root = output / "retained"
    records = {
        "tokens": retain_exact_npy(root / "tokens.npy", tokens.numpy()),
        "base": retain_exact_npy(root / "base_pre_r.npy", base.detach().numpy()),
        "zero": retain_exact_npy(root / "zero_pre_r.npy", zero.detach().numpy()),
        "nonzero": retain_exact_npy(root / "nonzero_pre_r.npy", moved.detach().numpy()),
    }
    result = {
        "schema": "ddm_ec2_toy_gate.v1",
        "scope": "STRUCTURAL_ONLY; synthetic 8x12 tokens; no scorer or empirical claim",
        "zero_identity": True,
        "nonzero_moves_receiver": True,
        "retained_payloads": records,
        "passed": True,
    }
    atomic_json(output / "TOY_GATE.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--resume-from", required=True)
    parser.add_argument("--toy-gate", type=Path)
    args = parser.parse_args()
    if args.toy_gate is not None:
        print(json.dumps(run_toy_gate(args.toy_gate.resolve()), indent=2, sort_keys=True))
        return
    if args.run_root is None:
        parser.error("--run-root is required outside --toy-gate")
    run_root = args.run_root.resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    request = json.loads((run_root / "inputs/REQUEST.json").read_text())
    if args.resume_from != request["run_id"]:
        raise EC2WorkerError("--resume-from must equal the immutable run_id")
    print(json.dumps(train(run_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
