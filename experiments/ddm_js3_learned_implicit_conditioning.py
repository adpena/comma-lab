#!/usr/bin/env python3
"""Resumable learned implicit-conditioning screen for the CP135 seg renderer.

The learned module consumes only receiver-known state (decoded semantic tokens,
their four-neighbour edge geometry, coordinates, and the frozen receiver RGB).
It emits a bounded correction before CP135's bilinear camera lift.  Every
candidate is then camera-uint8-STE rounded, bilinear-resized to the scorer grid,
transported onto the retained T4 custody plane, and evaluated by the frozen
CPU SegNet.  The local result is a relative gauge, never an exact score.

The runner is stage-resumable.  Each stage keeps the live and EMA checkpoints,
the exported/coded module variants, camera frames, logits, and argmax field.
It does not launch the charter's long burn or a paid evaluation.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import io
import itertools
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import brotli
import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments import ddm_js2b_edge_conditioning_relative_gauge as js2b

DEFAULT_OUTPUT = Path("/Volumes/VertigoDataTier/pact/ddm_js3_20260812")
AXIS = "[macOS-CPU advisory, instrument floor 0.0131 S]"
SEED = 20_260_812
BATCH = 16
THREADS = 8
DELTA = 0.08036041259765625
POSE_GUARD = 2e-6
T4_FLIP_GATE = -2_000
T4_BYTE_GATE = 1_500
RATE_DENOMINATOR = 37_545_489
H, W = 384, 512
CAMERA_H, CAMERA_W = 874, 1164
SAMPLE_N = 32
CHANNELS = 14
MODULE_MAGIC = b"JS3C\x01"
DEFAULT_STAGE_STEPS = (1, 4)
CAPACITY_LADDER = (4, 8, 12)


class JS3Error(RuntimeError):
    """A custody, resume, receiver-chain, or retention invariant failed."""


@dataclasses.dataclass(frozen=True)
class ExportedModule:
    mode: str
    raw: bytes
    coded: bytes
    decoded_state: dict[str, np.ndarray]
    report: dict[str, Any]


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


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


def atomic_torch_save(path: Path, value: Any, torch: Any) -> None:
    buffer = io.BytesIO()
    torch.save(value, buffer)
    atomic_bytes(path, buffer.getvalue())


def parse_stage_steps(value: str) -> tuple[int, ...]:
    stages = tuple(int(item) for item in value.split(",") if item.strip())
    if not stages or stages[0] <= 0 or any(b <= a for a, b in itertools.pairwise(stages)):
        raise argparse.ArgumentTypeError("stage steps must be strictly increasing positive integers")
    return stages


def stratified_sample() -> tuple[np.ndarray, np.ndarray]:
    return js2b.stratified_sample(seed=SEED, population=600, count=SAMPLE_N)


def fixed_context(torch: Any, functional: Any, tokens: Any, pre_r: Any) -> Any:
    """Return receiver-free context: classes, edges, xy, and frozen RGB."""
    one_hot = functional.one_hot(tokens.long(), num_classes=5).permute(0, 3, 1, 2).float()
    left = functional.pad(tokens[:, :, 1:] != tokens[:, :, :-1], (1, 0, 0, 0)).float()
    right = functional.pad(tokens[:, :, 1:] != tokens[:, :, :-1], (0, 1, 0, 0)).float()
    up = functional.pad(tokens[:, 1:, :] != tokens[:, :-1, :], (0, 0, 1, 0)).float()
    down = functional.pad(tokens[:, 1:, :] != tokens[:, :-1, :], (0, 0, 0, 1)).float()
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, H, dtype=pre_r.dtype, device=pre_r.device),
        torch.linspace(-1.0, 1.0, W, dtype=pre_r.dtype, device=pre_r.device),
        indexing="ij",
    )
    coordinates = torch.stack((xx, yy))[None].expand(tokens.shape[0], -1, -1, -1)
    rgb = pre_r / 127.5 - 1.0
    value = torch.cat((one_hot, left[:, None], right[:, None], up[:, None], down[:, None], coordinates, rgb), dim=1)
    if value.shape[1:] != (CHANNELS, H, W):
        raise JS3Error(f"implicit context geometry differs: {tuple(value.shape)}")
    return value


def fake_quant_weight(torch: Any, weight: Any) -> Any:
    scale = weight.detach().abs().amax().clamp_min(1e-8) / 127.0
    return torch.fake_quantize_per_tensor_affine(weight, float(scale), 0, -127, 127)


def build_model(torch: Any, functional: Any, hidden: int, max_delta: float, qat: bool = True) -> Any:
    if hidden not in CAPACITY_LADDER:
        raise ValueError(f"hidden must be one of {CAPACITY_LADDER}")

    class EdgeConditioner(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.hidden = hidden
            self.max_delta = max_delta
            self.qat = qat
            self.context = torch.nn.Conv2d(CHANNELS, hidden, 3, padding=1)
            self.depthwise = torch.nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
            self.head = torch.nn.Conv2d(hidden, 3, 1)
            torch.nn.init.kaiming_uniform_(self.context.weight, a=math.sqrt(5))
            torch.nn.init.zeros_(self.context.bias)
            torch.nn.init.dirac_(self.depthwise.weight)
            torch.nn.init.zeros_(self.depthwise.bias)
            torch.nn.init.zeros_(self.head.weight)
            torch.nn.init.zeros_(self.head.bias)

        def _conv(self, value: Any, layer: Any, *, groups: int = 1) -> Any:
            weight = fake_quant_weight(torch, layer.weight) if self.qat else layer.weight
            return functional.conv2d(value, weight, layer.bias, padding=layer.padding, groups=groups)

        def forward(self, value: Any) -> Any:
            value = functional.gelu(self._conv(value, self.context))
            value = functional.gelu(self._conv(value, self.depthwise, groups=self.hidden))
            return torch.tanh(self._conv(value, self.head)) * self.max_delta

    return EdgeConditioner()


def camera_roundtrip(torch: Any, functional: Any, pre_r: Any, correction: Any) -> tuple[Any, Any]:
    from tac.differentiable_eval_roundtrip import (
        CameraLiftKernel,
        EvalRoundTripOrdering,
        apply_camera_uint8_lift_during_training,
        apply_eval_roundtrip_during_training,
    )

    del functional  # The canonical HR2 apparatus owns both interpolation operations.
    value = pre_r + correction
    camera = apply_camera_uint8_lift_during_training(
        value,
        lift_kernel=CameraLiftKernel.BILINEAR,
        target_h=CAMERA_H,
        target_w=CAMERA_W,
    )
    scorer = apply_eval_roundtrip_during_training(
        value,
        ordering=EvalRoundTripOrdering.CAMERA_UINT8,
        lift_kernel=CameraLiftKernel.BILINEAR,
        target_h=CAMERA_H,
        target_w=CAMERA_W,
    )
    return camera, scorer


def signed_gt_margin(torch: Any, logits: Any, labels: Any) -> Any:
    target = logits.gather(1, labels[:, None]).squeeze(1)
    masked = logits.clone()
    masked.scatter_(1, labels[:, None], -torch.inf)
    return target - masked.amax(dim=1)


def delta_hinge_objective(torch: Any, logits: Any, labels: Any, baseline_pred: Any, delta: float = DELTA) -> tuple[Any, dict[str, Any]]:
    margin = signed_gt_margin(torch, logits, labels)
    hinge = torch.relu(delta - margin)
    wrong = baseline_pred != labels
    correct = ~wrong
    if not bool(wrong.any()):
        raise JS3Error("sample has no GT-disagreeing pixels")
    repair = hinge[wrong].mean()
    collateral = hinge[correct].mean()
    loss = repair + collateral
    return loss, {
        "repair_hinge": repair,
        "collateral_hinge": collateral,
        "signed_margin": margin,
    }


def robust_metrics(logits: np.ndarray, baseline: np.ndarray, gt: np.ndarray, weights: np.ndarray) -> dict[str, Any]:
    prediction = logits.argmax(axis=1).astype(np.uint8)
    index = np.arange(logits.shape[0])[:, None, None]
    rows = np.arange(H)[None, :, None]
    columns = np.arange(W)[None, None, :]
    target = logits[index, gt, rows, columns]
    masked = logits.copy()
    masked[index, gt, rows, columns] = -np.inf
    margin = target - masked.max(axis=1)
    base_correct = baseline == gt
    candidate_correct = prediction == gt
    beneficial = ~base_correct & candidate_correct
    harmful = base_correct & ~candidate_correct
    robust_beneficial = beneficial & (margin >= DELTA)
    robust_harmful = harmful & (margin <= -DELTA)
    delta_per_pair = (~candidate_correct).sum(axis=(1, 2)) - (~base_correct).sum(axis=(1, 2))
    robust_delta_per_pair = robust_harmful.sum(axis=(1, 2)) - robust_beneficial.sum(axis=(1, 2))
    projected_delta = js2b.projected_sum(delta_per_pair, weights)
    projected_robust = js2b.projected_sum(robust_delta_per_pair, weights)
    return {
        "prediction": prediction,
        "sample_delta_flips": int(delta_per_pair.sum()),
        "projected_n600_delta_flips": projected_delta,
        "sample_robust_delta_flips": int(robust_delta_per_pair.sum()),
        "projected_n600_robust_delta_flips": projected_robust,
        "beneficial_flips": int(beneficial.sum()),
        "harmful_flips": int(harmful.sum()),
        "robust_beneficial_flips": int(robust_beneficial.sum()),
        "robust_harmful_flips": int(robust_harmful.sum()),
        "tie_fragile_beneficial_flips": int((beneficial & (margin < DELTA)).sum()),
        "candidate_error_count_n32": int((~candidate_correct).sum()),
        "baseline_error_count_n32": int((~base_correct).sum()),
    }


def _tensor_records(model: Any) -> list[tuple[str, np.ndarray]]:
    return [(name, value.detach().cpu().numpy()) for name, value in sorted(model.state_dict().items())]


def serialize_module(model: Any, mode: str, retention_root: Path) -> ExportedModule:
    if mode not in {"fp16", "int8"}:
        raise ValueError("mode must be fp16 or int8")
    metadata: list[dict[str, Any]] = []
    chunks: list[bytes] = []
    decoded: dict[str, np.ndarray] = {}
    for name, array in _tensor_records(model):
        if mode == "fp16" or not name.endswith("weight"):
            stored = np.asarray(array, dtype="<f2")
            row = {"name": name, "shape": list(array.shape), "dtype": "float16", "scale": None}
            restored = stored.astype(np.float32)
        else:
            scale = max(float(np.max(np.abs(array))) / 127.0, 1e-8)
            stored = np.clip(np.rint(array / scale), -127, 127).astype(np.int8)
            row = {"name": name, "shape": list(array.shape), "dtype": "int8", "scale": scale}
            restored = stored.astype(np.float32) * scale
        raw = stored.tobytes(order="C")
        row["bytes"] = int(stored.size * stored.dtype.itemsize)
        metadata.append(row)
        chunks.append(raw)
        decoded[name] = restored.reshape(array.shape)
    header = json.dumps(
        {
            "schema": "ddm_js3_conditioner_payload.v1",
            "hidden": int(model.hidden),
            "max_delta": float(model.max_delta),
            "mode": mode,
            "tensors": metadata,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    raw = MODULE_MAGIC + len(header).to_bytes(4, "little") + header + b"".join(chunks)
    coded = brotli.compress(raw, quality=11)
    raw_path = retention_root / f"conditioner.{mode}.raw"
    coded_path = retention_root / f"conditioner.{mode}.br"
    atomic_bytes(raw_path, raw)
    atomic_bytes(coded_path, coded)
    restored = parse_module(coded)
    if set(restored) != set(decoded) or any(
        not np.array_equal(restored[name], decoded[name]) for name in decoded
    ):
        raise JS3Error(f"{mode} module failed Brotli parse-back")
    return ExportedModule(
        mode=mode,
        raw=raw,
        coded=coded,
        decoded_state=decoded,
        report={
            "mode": mode,
            "parameter_count": int(sum(value.size for _, value in _tensor_records(model))),
            "raw_bytes": raw_path.stat().st_size,
            "brotli_q11_bytes": coded_path.stat().st_size,
            "raw_sha256": sha256_bytes(raw),
            "brotli_q11_sha256": sha256_bytes(coded),
            "parseback_exact": True,
            "raw": file_record(raw_path),
            "coded": file_record(coded_path),
        },
    )


def parse_module(coded: bytes) -> dict[str, np.ndarray]:
    raw = brotli.decompress(coded)
    if not raw.startswith(MODULE_MAGIC) or len(raw) < len(MODULE_MAGIC) + 4:
        raise JS3Error("invalid learned-conditioner module")
    offset = len(MODULE_MAGIC)
    header_bytes = int.from_bytes(raw[offset : offset + 4], "little")
    offset += 4
    header = json.loads(raw[offset : offset + header_bytes])
    offset += header_bytes
    output: dict[str, np.ndarray] = {}
    for row in header["tensors"]:
        end = offset + int(row["bytes"])
        dtype = np.dtype("<f2") if row["dtype"] == "float16" else np.dtype(np.int8)
        value = np.frombuffer(raw[offset:end], dtype=dtype).copy().reshape(row["shape"])
        if row["dtype"] == "float16":
            value = value.astype(np.float32)
        else:
            value = value.astype(np.float32) * float(row["scale"])
        output[str(row["name"])] = value
        offset = end
    if offset != len(raw):
        raise JS3Error("learned-conditioner module has trailing bytes")
    return output


def load_decoded_state(model: Any, decoded: dict[str, np.ndarray], torch: Any) -> None:
    model.load_state_dict({name: torch.from_numpy(value.copy()) for name, value in decoded.items()}, strict=True)


def update_ema(ema: dict[str, Any], model: Any, decay: float) -> None:
    for name, value in model.state_dict().items():
        ema[name].mul_(decay).add_(value.detach(), alpha=1.0 - decay)


def pose_loss_for_batch(context: Any, camera: Any, indices: np.ndarray, base_pose_errors: np.ndarray) -> tuple[Any, float]:
    torch = context.modules.torch
    frame0 = torch.from_numpy(np.asarray(context.base_pairs[indices, 0]).copy()).permute(0, 3, 1, 2).float()
    pair = torch.stack((frame0, camera), dim=1)
    candidate_preprocessed = context.posenet.preprocess_input(pair)
    custody = torch.from_numpy(np.asarray(context.custody_pose[context.sample[indices]]).copy()).float()
    base = context.base_pose_input[indices]
    transported = custody + (candidate_preprocessed - base)
    output = context.posenet(transported)["pose"][..., :6]
    target = torch.from_numpy(np.asarray(context.gt_poses[context.sample[indices]], dtype=np.float32).copy())
    per_pair = (output - target).square().mean(dim=1)
    baseline = torch.from_numpy(np.asarray(base_pose_errors[indices], dtype=np.float32).copy())
    guard = torch.relu(per_pair - baseline).mean()
    return guard, float(per_pair.detach().mean())


def materialize_base_pre_r(context: Any, output: Path) -> np.ndarray:
    path = output / "inputs/retained/base_pre_r_n32.float16.npy"
    if path.is_file():
        value = np.load(path, mmap_mode="r", allow_pickle=False)
        if value.shape != (SAMPLE_N, 3, H, W) or value.dtype != np.float16:
            raise JS3Error("retained base pre-R render differs")
        return value
    torch = context.modules.torch
    renderer = context.modules.renderer_runtime.SemanticTokenRenderer(96).eval()
    state = {
        record.schema.name: torch.from_numpy(np.ascontiguousarray(record.values, dtype=np.float32))
        for record in context.records
    }
    renderer.load_state_dict(state, strict=True)
    tokens = torch.from_numpy(np.asarray(context.tokens[context.sample]).astype(np.int64, copy=True))
    indices = torch.from_numpy(context.sample.copy()).long()
    chunks = []
    with torch.inference_mode():
        for start in range(0, SAMPLE_N, BATCH):
            chunks.append(renderer(tokens[start : start + BATCH], indices[start : start + BATCH]).half().cpu().numpy())
    value = np.concatenate(chunks)
    atomic_npy(path, value)
    return np.load(path, mmap_mode="r", allow_pickle=False)


def evaluate_model(context: Any, model: Any, pre_r_store: np.ndarray, output: Path, label: str) -> dict[str, Any]:
    torch, functional = context.modules.torch, context.modules.functional
    logits_rows: list[np.ndarray] = []
    camera_rows: list[np.ndarray] = []
    correction_rows: list[np.ndarray] = []
    pose_errors: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, SAMPLE_N, BATCH):
            stop = start + BATCH
            pre_r = torch.from_numpy(np.asarray(pre_r_store[start:stop], dtype=np.float32))
            tokens = torch.from_numpy(np.asarray(context.tokens[context.sample[start:stop]]).astype(np.int64, copy=True))
            correction = model(fixed_context(torch, functional, tokens, pre_r))
            camera, scorer = camera_roundtrip(torch, functional, pre_r, correction)
            base_camera, base_scorer = camera_roundtrip(torch, functional, pre_r, torch.zeros_like(correction))
            custody = torch.from_numpy(np.asarray(context.custody_seg[context.sample[start:stop]]).copy()).float()
            transported = custody + (scorer - base_scorer)
            logits_rows.append(context.segnet(transported).cpu().numpy().astype(np.float32))
            camera_rows.append(camera.permute(0, 2, 3, 1).to(torch.uint8).cpu().numpy())
            correction_rows.append(correction.half().cpu().numpy())
            frame0 = torch.from_numpy(np.asarray(context.base_pairs[start:stop, 0]).copy()).permute(0, 3, 1, 2).float()
            pair = torch.stack((frame0, camera), dim=1)
            candidate_preprocessed = context.posenet.preprocess_input(pair)
            custody_pose = torch.from_numpy(np.asarray(context.custody_pose[context.sample[start:stop]]).copy()).float()
            transported_pose = custody_pose + (candidate_preprocessed - context.base_pose_input[start:stop])
            pose = context.posenet(transported_pose)["pose"][..., :6].cpu().numpy()
            pose_errors.append(((pose.astype(np.float64) - context.gt_poses[context.sample[start:stop]]) ** 2).mean(axis=1))
    logits = np.concatenate(logits_rows)
    camera = np.concatenate(camera_rows)
    correction = np.concatenate(correction_rows)
    errors = np.concatenate(pose_errors)
    gt = np.asarray(context.gt_labels[context.sample])
    baseline = np.asarray(context.custody_argmax[context.sample])
    metrics = robust_metrics(logits, baseline, gt, context.sample_weights)
    prediction = metrics.pop("prediction")
    root = output / "stages" / label / "retained"
    atomic_npy(root / "camera_frame1_n32.uint8.npy", camera)
    atomic_npy(root / "correction_n32.float16.npy", correction)
    atomic_npy(root / "logits_n32.float32.npy", logits)
    atomic_npy(root / "argmax_n32.uint8.npy", prediction)
    atomic_npy(root / "pose_errors_n32.float64.npy", errors)
    base_pose = js2b.stratified_mean(context.base_pose_errors, context.sample_weights)
    candidate_pose = js2b.stratified_mean(errors, context.sample_weights)
    metrics.update(
        {
            "pose_base_stratified_n32": base_pose,
            "pose_candidate_stratified_n32": candidate_pose,
            "pose_delta_stratified_n32": candidate_pose - base_pose,
            "pose_guard_pass": candidate_pose - base_pose < POSE_GUARD,
            "payloads": {
                "camera": file_record(root / "camera_frame1_n32.uint8.npy"),
                "correction": file_record(root / "correction_n32.float16.npy"),
                "logits": file_record(root / "logits_n32.float32.npy"),
                "argmax": file_record(root / "argmax_n32.uint8.npy"),
                "pose_errors": file_record(root / "pose_errors_n32.float64.npy"),
            },
        }
    )
    return metrics


def export_stage(model: Any, output: Path, label: str) -> dict[str, Any]:
    root = output / "stages" / label / "retained"
    rows = []
    exports = []
    for mode in ("fp16", "int8"):
        exported = serialize_module(model, mode, root)
        row = dict(exported.report)
        rows.append(row)
        exports.append(exported)
    selected = min(rows, key=lambda row: (int(row["brotli_q11_bytes"]), str(row["mode"])))
    return {"races": rows, "selected": selected, "exports": exports}


def save_checkpoint(
    model: Any,
    optimizer: Any,
    ema: dict[str, Any],
    output: Path,
    label: str,
    step: int,
    history: list[dict[str, Any]],
    torch: Any,
    run_config: dict[str, Any],
) -> dict[str, Any]:
    path = output / "checkpoints" / f"{label}.step_{step:06d}.pt"
    payload = {
        "schema": "ddm_js3_checkpoint.v1",
        "step": step,
        "stage": label,
        "model": model.state_dict(),
        "ema": ema,
        "optimizer": optimizer.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "numpy_rng": np.random.get_state(),
        "history": history,
        "config": run_config,
    }
    atomic_torch_save(path, payload, torch)
    latest = {"schema": "ddm_js3_latest_checkpoint.v1", "checkpoint": file_record(path), "step": step, "stage": label}
    atomic_json(output / "checkpoints/LATEST.json", latest)
    return latest


def capacity_ladder(torch: Any, functional: Any, output: Path, max_delta: float) -> list[dict[str, Any]]:
    rows = []
    for hidden in CAPACITY_LADDER:
        torch.manual_seed(SEED + hidden)
        model = build_model(torch, functional, hidden, max_delta, qat=True)
        retention_root = output / "design/retained" / f"hidden_{hidden}"
        race = [
            serialize_module(model, mode, retention_root).report
            for mode in ("fp16", "int8")
        ]
        selected = min(race, key=lambda row: (int(row["brotli_q11_bytes"]), str(row["mode"])))
        rows.append({"hidden": hidden, "parameter_count": selected["parameter_count"], "races": race, "selected": selected})
    atomic_json(output / "design/CAPACITY_LADDER.json", {"schema": "ddm_js3_capacity_ladder.v1", "rows": rows})
    return rows


def memory_preflight(output: Path, hidden: int) -> dict[str, Any]:
    page_size = 16_384
    free_pages = None
    try:
        import subprocess

        completed = subprocess.run(["vm_stat"], check=True, capture_output=True, text=True)
        first = completed.stdout.splitlines()[0]
        page_size = int(first.split("page size of ", 1)[1].split(" bytes", 1)[0])
        for line in completed.stdout.splitlines():
            if line.startswith("Pages free:"):
                free_pages = int(line.split(":", 1)[1].strip().rstrip("."))
                break
    except (OSError, ValueError, IndexError):
        free_pages = None
    estimated = BATCH * (CHANNELS + 3 * 7 + hidden * 6) * H * W * 4
    row = {
        "schema": "ddm_js3_memory_preflight.v1",
        "platform": platform.platform(),
        "batch": BATCH,
        "threads": THREADS,
        "hidden": hidden,
        "estimated_working_set_bytes_lower_bound": estimated,
        "free_memory_bytes_observed": None if free_pages is None else free_pages * page_size,
        "pass": free_pages is None or free_pages * page_size >= 3 * estimated,
        "rule": "fail closed when observed free memory is below 3x the analytical lower bound",
    }
    atomic_json(output / "preflight/MEMORY_PREFLIGHT.json", row)
    if not row["pass"]:
        raise JS3Error("memory preflight failed")
    return row


def run_config(args: argparse.Namespace) -> dict[str, Any]:
    config = {
        "seed": SEED,
        "batch": BATCH,
        "threads": THREADS,
        "hidden": args.hidden,
        "max_delta": args.max_delta,
        "delta": DELTA,
        "lr": args.lr,
        "pose_every": args.pose_every,
        "pose_weight": args.pose_weight,
        "ema_decay": args.ema_decay,
        "grad_clip": args.grad_clip,
        "qat": True,
    }
    target_bindings = getattr(args, "target_bindings", None)
    if target_bindings is not None:
        config["target_bindings"] = target_bindings
    return config


def load_or_start(args: argparse.Namespace, torch: Any, functional: Any) -> tuple[Any, Any, dict[str, Any], int, list[dict[str, Any]]]:
    model = build_model(torch, functional, args.hidden, args.max_delta, qat=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    ema = {name: value.detach().clone() for name, value in model.state_dict().items()}
    start_step = 0
    history: list[dict[str, Any]] = []
    latest_path = args.output / "checkpoints/LATEST.json"
    if args.resume and latest_path.is_file():
        latest = json.loads(latest_path.read_text())
        checkpoint = torch.load(latest["checkpoint"]["path"], map_location="cpu", weights_only=False)
        config = checkpoint["config"]
        if config != run_config(args):
            raise JS3Error("resume checkpoint config differs")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        ema = checkpoint["ema"]
        torch.set_rng_state(checkpoint["torch_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
        history = list(checkpoint["history"])
        start_step = int(checkpoint["step"])
    return model, optimizer, ema, start_step, history


def train(args: argparse.Namespace, context: Any, pre_r_store: np.ndarray) -> dict[str, Any]:
    torch, functional = context.modules.torch, context.modules.functional
    model, optimizer, ema, start_step, history = load_or_start(args, torch, functional)
    config = run_config(args)
    model.train()
    gt_all = np.asarray(context.gt_labels[context.sample])
    baseline_all = np.asarray(context.custody_argmax[context.sample])
    run_started = time.perf_counter()
    training_seconds = 0.0
    history_at_start = len(history)
    completed_stages: list[dict[str, Any]] = []
    for stage_index, stage_end in enumerate(args.stage_steps, start=1):
        label = f"stage_{stage_index:02d}_step_{stage_end:06d}"
        if stage_end <= start_step:
            result_path = args.output / "stages" / label / "RESULT.json"
            if not result_path.is_file():
                raise JS3Error(f"resume checkpoint passed a stage with no result: {result_path}")
            retained_stage = json.loads(result_path.read_text())
            if retained_stage.get("stage") != label or int(retained_stage.get("step", -1)) != stage_end:
                raise JS3Error(f"retained stage result identity differs: {result_path}")
            completed_stages.append(retained_stage)
            continue
        stage_started = time.perf_counter()
        for step in range(start_step + 1, stage_end + 1):
            batch_slot = (step - 1) % (SAMPLE_N // BATCH)
            start = batch_slot * BATCH
            stop = start + BATCH
            pre_r = torch.from_numpy(np.asarray(pre_r_store[start:stop], dtype=np.float32))
            tokens = torch.from_numpy(np.asarray(context.tokens[context.sample[start:stop]]).astype(np.int64, copy=True))
            gt = torch.from_numpy(gt_all[start:stop].astype(np.int64, copy=True))
            baseline = torch.from_numpy(baseline_all[start:stop].astype(np.int64, copy=True))
            correction = model(fixed_context(torch, functional, tokens, pre_r))
            camera, scorer = camera_roundtrip(torch, functional, pre_r, correction)
            with torch.no_grad():
                _, base_scorer = camera_roundtrip(torch, functional, pre_r, torch.zeros_like(correction))
            custody = torch.from_numpy(np.asarray(context.custody_seg[context.sample[start:stop]]).copy()).float()
            logits = context.segnet(custody + (scorer - base_scorer))
            seg_loss, terms = delta_hinge_objective(torch, logits, gt, baseline)
            pose_guard = torch.zeros((), dtype=seg_loss.dtype)
            pose_value = None
            if args.pose_every > 0 and step % args.pose_every == 0:
                pose_guard, pose_value = pose_loss_for_batch(
                    context,
                    camera,
                    np.arange(start, stop),
                    context.base_pose_errors,
                )
            loss = seg_loss + args.pose_weight * pose_guard
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            update_ema(ema, model, args.ema_decay)
            row = {
                "step": step,
                "loss": float(loss.detach()),
                "repair_hinge": float(terms["repair_hinge"].detach()),
                "collateral_hinge": float(terms["collateral_hinge"].detach()),
                "pose_guard_loss": float(pose_guard.detach()),
                "pose_mse_batch": pose_value,
                "batch_start": start,
                "batch_size": BATCH,
            }
            history.append(row)
            start_step = step
            if step % args.checkpoint_every == 0 and step != stage_end:
                save_checkpoint(model, optimizer, ema, args.output, "periodic", step, history, torch, config)
            if time.perf_counter() - run_started > args.max_wall_seconds:
                save_checkpoint(model, optimizer, ema, args.output, "wall_stop", step, history, torch, config)
                raise JS3Error("bounded smoke wall-clock cap reached; checkpoint retained")
        training_seconds += time.perf_counter() - stage_started
        checkpoint = save_checkpoint(model, optimizer, ema, args.output, label, stage_end, history, torch, config)
        live_metrics = evaluate_model(context, model, pre_r_store, args.output, label)
        export = export_stage(model, args.output, label)
        live_selected_export = next(
            item for item in export["exports"] if item.mode == export["selected"]["mode"]
        )
        live_coded_model = build_model(torch, functional, args.hidden, args.max_delta, qat=True)
        load_decoded_state(live_coded_model, live_selected_export.decoded_state, torch)
        live_coded_metrics = evaluate_model(
            context,
            live_coded_model,
            pre_r_store,
            args.output,
            label + "_coded",
        )
        ema_model = build_model(torch, functional, args.hidden, args.max_delta, qat=True)
        ema_model.load_state_dict(ema)
        ema_label = label + "_ema"
        ema_metrics = evaluate_model(context, ema_model, pre_r_store, args.output, ema_label)
        ema_export = export_stage(ema_model, args.output, ema_label)
        ema_selected_export = next(
            item for item in ema_export["exports"] if item.mode == ema_export["selected"]["mode"]
        )
        ema_coded_model = build_model(torch, functional, args.hidden, args.max_delta, qat=True)
        load_decoded_state(ema_coded_model, ema_selected_export.decoded_state, torch)
        ema_coded_metrics = evaluate_model(
            context,
            ema_coded_model,
            pre_r_store,
            args.output,
            ema_label + "_coded",
        )
        selected_kind, selected_metrics, selected_export = min(
            (("live", live_coded_metrics, export), ("ema", ema_coded_metrics, ema_export)),
            key=lambda item: (
                not bool(item[1]["pose_guard_pass"]),
                int(item[1]["projected_n600_robust_delta_flips"]),
                int(item[1]["projected_n600_delta_flips"]),
                int(item[2]["selected"]["brotli_q11_bytes"]),
                item[0],
            ),
        )
        row = {
            "stage": label,
            "step": stage_end,
            "checkpoint": checkpoint,
            "live": {
                "float_qat_metrics": live_metrics,
                "coded_metrics": live_coded_metrics,
                "export": {k: v for k, v in export.items() if k != "exports"},
            },
            "ema": {
                "float_qat_metrics": ema_metrics,
                "coded_metrics": ema_coded_metrics,
                "export": {k: v for k, v in ema_export.items() if k != "exports"},
            },
            "selected_kind": selected_kind,
            "selected_metrics": selected_metrics,
            "selected_export": {k: v for k, v in selected_export.items() if k != "exports"},
        }
        atomic_json(args.output / "stages" / label / "RESULT.json", row)
        completed_stages.append(row)
    elapsed = time.perf_counter() - run_started
    return {
        "model": model,
        "ema": ema,
        "history": history,
        "stages": completed_stages,
        "steps_completed": start_step,
        "training_seconds": training_seconds,
        "wall_seconds": elapsed,
        "seconds_per_step": training_seconds / max(1, len(history) - history_at_start),
    }


def sealed_recipe(args: argparse.Namespace, seconds_per_step: float, selected_bytes: int, output: Path) -> dict[str, Any]:
    projected_300 = seconds_per_step * 300
    recipe = {
        "schema": "ddm_js3_sealed_main_recipe.v1",
        "disposition": "QUEUED-WITH-A-FIRE-ORDER",
        "owner": "MAIN training-leg router",
        "consumer_store": str(output / "main_burn"),
        "fire_trigger": (
            "MAIN owns the training leg; memory preflight passes on the target host; no other full-n600 scorer job is active; "
            "consume js2b ROUTE.json and do not continue the two-W4 catalog"
        ),
        "command": [
            ".venv/bin/python",
            "experiments/ddm_js3_learned_implicit_conditioning.py",
            "--output",
            str(output / "main_burn"),
            "--hidden",
            str(args.hidden),
            "--max-delta",
            str(args.max_delta),
            "--lr",
            str(args.lr),
            "--stage-steps",
            "100,300,900",
            "--checkpoint-every",
            "25",
            "--pose-every",
            str(args.pose_every),
            "--pose-weight",
            str(args.pose_weight),
            "--ema-decay",
            str(args.ema_decay),
            "--grad-clip",
            str(args.grad_clip),
            "--max-wall-seconds",
            str(max(7200, math.ceil(seconds_per_step * 950))),
            "--resume",
        ],
        "seed": SEED,
        "sample_n": SAMPLE_N,
        "sample_policy": "seeded stratified random, one draw per 600/32 stratum",
        "batch": BATCH,
        "threads": THREADS,
        "delta": DELTA,
        "capacity_hidden": args.hidden,
        "latest_selected_module_brotli_q11_bytes": selected_bytes,
        "projected_300_step_seconds_from_smoke": projected_300,
        "projected_300_step_minutes_from_smoke": projected_300 / 60.0,
        "t4_admission": {
            "projected_n600_robust_delta_flips_lte": T4_FLIP_GATE,
            "module_brotli_q11_bytes_lte": T4_BYTE_GATE,
            "pose_delta_lt": POSE_GUARD,
            "requires_complete_receiver_consumption_and_n600_local_instrument_before_dispatch": True,
        },
        "long_burn_launched_by_ddm_js3": False,
    }
    atomic_json(output / "SEALED_MAIN_RECIPE.json", recipe)
    return recipe


def finalize(args: argparse.Namespace, training: dict[str, Any], ladder: list[dict[str, Any]], preflight: dict[str, Any]) -> dict[str, Any]:
    if not training["stages"]:
        raise JS3Error("no new stage completed")
    selected_stage = min(
        training["stages"],
        key=lambda row: (
            not bool(row["selected_metrics"]["pose_guard_pass"]),
            int(row["selected_metrics"]["projected_n600_robust_delta_flips"]),
            int(row["selected_metrics"]["projected_n600_delta_flips"]),
            int(row["selected_export"]["selected"]["brotli_q11_bytes"]),
            int(row["step"]),
        ),
    )
    metrics = selected_stage["selected_metrics"]
    module_bytes = int(selected_stage["selected_export"]["selected"]["brotli_q11_bytes"])
    projected_robust = int(metrics["projected_n600_robust_delta_flips"])
    pose_delta = float(metrics["pose_delta_stratified_n32"])
    gate = projected_robust <= T4_FLIP_GATE and module_bytes <= T4_BYTE_GATE and pose_delta < POSE_GUARD
    recipe = sealed_recipe(args, float(training["seconds_per_step"]), module_bytes, args.output)
    hinge_start = float(training["history"][0]["repair_hinge"])
    hinge_end = float(training["history"][-1]["repair_hinge"])
    f1_eligible = training["steps_completed"] >= 300
    f1_fired = f1_eligible and projected_robust == 0 and hinge_end < hinge_start
    bytes_per_robust = module_bytes / -projected_robust if projected_robust < 0 else None
    f2_eligible = False
    f2_fired = False
    follow_on = {
        "disposition": "QUEUED-WITH-A-FIRE-ORDER",
        "owner": "MAIN training-leg router",
        "consumer_store": recipe["consumer_store"],
        "fire_trigger": recipe["fire_trigger"] if not gate else (
            "MAIN verifies receiver consumption and an n600 local-instrument projection preserves the n32 gate; then claim the sole T4 lane"
        ),
        "bounded_smoke_gate_pass": gate,
    }
    result = {
        "schema": "ddm_js3_final_result.v1",
        "axis": AXIS,
        "score_claim": False,
        "sample": stratified_sample()[0].tolist(),
        "sample_weights": stratified_sample()[1].tolist(),
        "steps_completed": training["steps_completed"],
        "seconds_per_step": training["seconds_per_step"],
        "selected_stage": selected_stage,
        "capacity_ladder": ladder,
        "memory_preflight": preflight,
        "t4_acceptance_gate": {
            "pass": gate,
            "projected_n600_robust_delta_flips": projected_robust,
            "required_projected_n600_robust_delta_flips_lte": T4_FLIP_GATE,
            "module_brotli_q11_bytes": module_bytes,
            "required_module_brotli_q11_bytes_lte": T4_BYTE_GATE,
            "pose_delta": pose_delta,
            "required_pose_delta_lt": POSE_GUARD,
            "dispatched": False,
        },
        "existence_signal": {
            "robust_gradient_movement_observed": projected_robust < 0,
            "projected_n600_robust_delta_flips": projected_robust,
            "projected_n600_total_delta_flips": metrics["projected_n600_delta_flips"],
            "pose_guard_pass": metrics["pose_guard_pass"],
            "admissible_candidate": gate,
        },
        "falsifiers": {
            "F1": {
                "eligible": f1_eligible,
                "fired": f1_fired,
                "scope": "INSTANCE: selected hidden-width rung on stratified-random n32 through real CP135 R/uint8 chain",
                "reason": None if f1_eligible else "fewer than 300 steps; bounded smoke cannot close the family",
            },
            "F2": {
                "eligible": f2_eligible,
                "fired": f2_fired,
                "scope": "not eligible: all capacity rungs were priced, but only hidden=4 was trained",
                "bytes_per_projected_robust_flip": bytes_per_robust,
                "hidden4_linearized_bytes_for_2000_robust_flips": (
                    None if bytes_per_robust is None else 2000.0 * bytes_per_robust
                ),
            },
            "F3": {
                "fired": False,
                "scope": "not eligible: every capacity rung was not trained",
                "selected_pose_guard_pass": pose_delta < POSE_GUARD,
            },
        },
        "follow_on": follow_on,
        "sealed_recipe": file_record(args.output / "SEALED_MAIN_RECIPE.json"),
        "boundaries": {
            "absolute_local_dseg_progress_claim": False,
            "exact_cuda_score_measured": False,
            "long_burn_launched": False,
            "candidate_archive_built": False,
            "receiver_module_integration_complete": False,
            "verdict_scope": "bounded n32 existence smoke and real module coder; not a formulation verdict below 300 steps",
        },
    }
    atomic_json(args.output / "FINAL_RESULT.json", result)
    atomic_json(args.output / "T4_ACCEPTANCE_FIRE_ORDER.json", follow_on)
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    torch = __import__("torch")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(THREADS)
    torch.use_deterministic_algorithms(True)
    state = {
        "schema": "ddm_js3_state.v1",
        "arm": "ddm_js3",
        "status": "CUSTODY_PREFLIGHT",
        "complete": False,
        "resumable": True,
        "seed": SEED,
        "batch": BATCH,
        "threads": THREADS,
        "axis": AXIS,
        "score_claim": False,
        "long_burn_launched": False,
    }
    atomic_json(args.output / "state.json", state)
    preflight = memory_preflight(args.output, args.hidden)
    context = js2b.build_context(args.output)
    if not np.array_equal(context.sample, stratified_sample()[0]):
        raise JS3Error("sample differs from js2b's sealed relative gauge")
    pre_r_store = materialize_base_pre_r(context, args.output)
    ladder = capacity_ladder(torch, context.modules.functional, args.output, args.max_delta)
    state.update(status="BOUNDED_TRAINING", memory_preflight=preflight)
    atomic_json(args.output / "state.json", state)
    training = train(args, context, pre_r_store)
    result = finalize(args, training, ladder, preflight)
    state.update(
        status="COMPLETE",
        complete=True,
        steps_completed=training["steps_completed"],
        final_result=file_record(args.output / "FINAL_RESULT.json"),
        t4_gate_pass=result["t4_acceptance_gate"]["pass"],
    )
    atomic_json(args.output / "state.json", state)
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    value.add_argument("--hidden", type=int, choices=CAPACITY_LADDER, default=4)
    value.add_argument("--max-delta", type=float, default=6.0)
    value.add_argument("--lr", type=float, default=0.02)
    value.add_argument("--stage-steps", type=parse_stage_steps, default=DEFAULT_STAGE_STEPS)
    value.add_argument("--checkpoint-every", type=int, default=1)
    value.add_argument("--pose-every", type=int, default=2)
    value.add_argument("--pose-weight", type=float, default=50.0)
    value.add_argument("--ema-decay", type=float, default=0.99)
    value.add_argument("--grad-clip", type=float, default=5.0)
    value.add_argument("--max-wall-seconds", type=float, default=1_800.0)
    value.add_argument("--resume", action="store_true")
    # Callers (e.g. ddm_sa1) supply target_bindings via a hand-built
    # Namespace, not a CLI flag; declare the default so the attribute
    # always exists on args.
    value.set_defaults(target_bindings=None)
    return value


def main() -> None:
    args = parser().parse_args()
    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
