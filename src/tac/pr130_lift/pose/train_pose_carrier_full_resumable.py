#!/usr/bin/env python3
# borrowed_substrate_accounting:
#   source_repo: /Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo
#   source_repo_head: e34f31bc4969042c0051ac81aa3c56884419a231
#   lifted_at_head: 2f94596bb0136d342254022a5c9584756eae0468
#   source_path: code/train_pose_carrier_full.py
#   theirs: PR130 pose carrier model, losses, quantization, optimizer semantics,
#     cache schema, and full-run defaults.
#   ours: wrapper-only full-state resume sidecar, smoke-pair scope reduction,
#     explicit optimizer-mode selection, execution provenance, and typed
#     checkpoint-metadata reads; the vendored trainer has only its separately
#     declared governed-admission adaptation.
"""Resumable wrapper for PR130's full pose-carrier trainer.

The vendored PR130 trainer saves deployable weights only.  This wrapper mirrors
the training mechanics while preserving optimizer, scheduler, RNG, sampling,
history, and best-state data in a sidecar checkpoint so long Row-2 runs can
resume without changing the borrowed pose-carrier mechanism.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tac.admission_guard import assert_governed_admission
from tac.pr130_lift.checkpoint_schema import architecture_config_from_checkpoint
from tac.pr130_lift.pose.mps_port import (
    DENSE_ADAPTER_MODE,
    REFERENCE_SPARSE_MODE,
    RowLocalDenseAdam,
    build_row_local_coefficients,
    clear_device_cache,
    load_safetensors_cpu_then_move,
    prepare_row_local_step,
    torch_public_version,
)
from tac.pr130_lift.pose.source_loader import lifted_script_path, load_lifted_module

N_TOTAL_PAIRS = 600
REPO_ROOT = Path(__file__).resolve().parents[4]
NATIVE_SPARSE_RECEIPT_PATH = Path(
    "/Volumes/VertigoDataTier/pact/ddm_pq1_probe_20260809/"
    "probe_torch2100_pinned.json"
)
NATIVE_SPARSE_RECEIPT_SHA256 = (
    "32ce0585d070fd578bea563f94b33fffe6e000b8cc608f827d4fcb5319893ec3"
)


def _jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def _native_probe_receipt_identity() -> dict[str, Any]:
    observed_sha256 = None
    status = "missing_at_run"
    if NATIVE_SPARSE_RECEIPT_PATH.is_file():
        observed_sha256 = hashlib.sha256(
            NATIVE_SPARSE_RECEIPT_PATH.read_bytes()
        ).hexdigest()
        status = (
            "verified_at_run"
            if observed_sha256 == NATIVE_SPARSE_RECEIPT_SHA256
            else "hash_mismatch"
        )
    if status == "hash_mismatch":
        raise RuntimeError(
            "native sparse probe receipt hash mismatch: "
            f"{NATIVE_SPARSE_RECEIPT_PATH}"
        )
    return {
        "path": str(NATIVE_SPARSE_RECEIPT_PATH),
        "expected_sha256": NATIVE_SPARSE_RECEIPT_SHA256,
        "observed_sha256": observed_sha256,
        "status": status,
        "scope": "Torch 2.10.0, two steps, four coefficient rows, real MPS",
        "score_claim": False,
    }


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()


def _execution_provenance(
    *,
    args: argparse.Namespace,
    coefficients: torch.nn.Embedding,
    optimizer: torch.optim.Optimizer,
    argv: list[str],
) -> dict[str, Any]:
    sparse = bool(coefficients.sparse)
    is_dense_adapter = isinstance(optimizer, RowLocalDenseAdam)
    if sparse == is_dense_adapter:
        raise RuntimeError("row-local optimizer class does not match gradient mode")
    expected_mode = (
        DENSE_ADAPTER_MODE if is_dense_adapter else REFERENCE_SPARSE_MODE
    )
    if args.row_local_mode != expected_mode:
        raise RuntimeError("row-local optimizer selection drifted from requested mode")
    return {
        "schema": "ddm_fx2_pose_optimizer_provenance.v1",
        "score_claim": False,
        "device": str(args.device),
        "optimizer_class": (
            f"{type(optimizer).__module__}.{type(optimizer).__qualname__}"
        ),
        "row_local_mode": expected_mode,
        "gradient_representation": "sparse" if sparse else "dense",
        "selection_event": (
            "reference_default"
            if expected_mode == REFERENCE_SPARSE_MODE
            else "explicit_dense_adapter_opt_in"
        ),
        "fallback_event": "none",
        "fallback_policy": "automatic_fallback_forbidden",
        "torch_version": torch_public_version(),
        "git_sha": _git_head(),
        "argv": list(argv),
        "native_probe_receipt": _native_probe_receipt_identity(),
    }


def _state_path(save: Path, step: int) -> Path:
    return save.with_name(f"{save.stem}.step{step:06d}.full_state.pt")


def _latest_state_path(save: Path) -> Path:
    return save.with_name(f"{save.stem}.full_state.latest.pt")


def _atomic_torch_save(payload: Any, path: Path) -> None:
    """Write a Torch artifact atomically in its destination directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _save_full_state(
    *,
    args: argparse.Namespace,
    step: int,
    raw_basis: torch.nn.Parameter,
    coeff: torch.nn.Embedding,
    basis_optimizer: torch.optim.Optimizer,
    coeff_optimizer: torch.optim.Optimizer,
    basis_scheduler: torch.optim.lr_scheduler.LRScheduler,
    coeff_scheduler: torch.optim.lr_scheduler.LRScheduler,
    generator: torch.Generator,
    order: torch.Tensor,
    cursor: int,
    sampling_weights: torch.Tensor | None,
    history: list[dict[str, Any]],
    best: dict[str, Any],
    active_pair_ids: torch.Tensor,
    execution_provenance: dict[str, Any],
) -> Path:
    payload = {
        "schema": "ddm_mx2_pose_carrier_full_state.v1",
        "step": int(step),
        "args": _jsonable_args(args),
        "raw_basis": raw_basis.detach().cpu().clone(),
        "coeff_weight": coeff.weight.detach().cpu().clone(),
        "basis_optimizer": basis_optimizer.state_dict(),
        "coeff_optimizer": coeff_optimizer.state_dict(),
        "basis_scheduler": basis_scheduler.state_dict(),
        "coeff_scheduler": coeff_scheduler.state_dict(),
        "generator_state": generator.get_state(),
        "order": order.detach().cpu().clone(),
        "cursor": int(cursor),
        "sampling_weights": (
            sampling_weights.detach().cpu().clone()
            if sampling_weights is not None else None
        ),
        "history": list(history),
        "best": {
            "mean": float(best["mean"]),
            "basis": (
                best["basis"].detach().cpu().clone()
                if best.get("basis") is not None else None
            ),
            "coeff": (
                best["coeff"].detach().cpu().clone()
                if best.get("coeff") is not None else None
            ),
        },
        "active_pair_ids": active_pair_ids.detach().cpu().tolist(),
        "execution_provenance": execution_provenance,
    }
    path = _state_path(args.save, step)
    latest = _latest_state_path(args.save)
    _atomic_torch_save(payload, path)
    _atomic_torch_save(payload, latest)
    print(json.dumps({
        "event": "full_state_saved",
        "step": step,
        "path": str(path),
        "latest": str(latest),
    }), flush=True)
    return path


def _check_resume_compatibility(
    payload: dict[str, Any],
    args: argparse.Namespace,
    active_pair_ids: torch.Tensor,
) -> None:
    if payload.get("schema") != "ddm_mx2_pose_carrier_full_state.v1":
        raise ValueError("resume-state schema mismatch")
    previous_args = payload.get("args", {})
    stable_keys = (
        "target_cache",
        "master_checkpoint",
        "init_carrier",
        "master_cache",
        "reuse_master_cache",
        "cache_masters_on_device",
        "steps",
        "batch_size",
        "eval_batch_size",
        "lr_basis",
        "lr_coeff",
        "basis_freeze_fraction",
        "basis_train_until_fraction",
        "qat_fraction",
        "coeff_qat_fraction",
        "metric_loss_after_basis",
        "always_metric_loss",
        "metric_normalized_weight",
        "hard_mining_power",
        "hard_mining_max",
        "basis_bits",
        "coeff_bits",
        "amplitude",
        "master_carrier_amplitude",
        "carrier_base",
        "zero_init_coeff",
        "seed",
        "device",
        "smoke_pairs",
    )
    current = _jsonable_args(args)
    for key in stable_keys:
        if previous_args.get(key) != current.get(key):
            raise ValueError(
                f"resume-state config mismatch for {key}: "
                f"{previous_args.get(key)!r} != {current.get(key)!r}"
            )
    previous_mode = previous_args.get("row_local_mode")
    if previous_mode is None:
        previous_device_type = torch.device(previous_args.get("device", "cpu")).type
        previous_mode = (
            DENSE_ADAPTER_MODE
            if previous_device_type == "mps"
            else REFERENCE_SPARSE_MODE
        )
    if previous_mode != current.get("row_local_mode"):
        raise ValueError(
            "resume-state row-local optimizer mode mismatch: "
            f"{previous_mode!r} != {current.get('row_local_mode')!r}"
        )
    if payload.get("active_pair_ids") != active_pair_ids.detach().cpu().tolist():
        raise ValueError("resume-state active pair ids mismatch")


def _load_full_state(
    path: Path,
    *,
    args: argparse.Namespace,
    raw_basis: torch.nn.Parameter,
    coeff: torch.nn.Embedding,
    basis_optimizer: torch.optim.Optimizer,
    coeff_optimizer: torch.optim.Optimizer,
    basis_scheduler: torch.optim.lr_scheduler.LRScheduler,
    coeff_scheduler: torch.optim.lr_scheduler.LRScheduler,
    generator: torch.Generator,
    active_pair_ids: torch.Tensor,
    device: torch.device,
) -> tuple[int, torch.Tensor, int, torch.Tensor | None, list[dict[str, Any]], dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    _check_resume_compatibility(payload, args, active_pair_ids)
    if tuple(payload["raw_basis"].shape) != tuple(raw_basis.shape):
        raise ValueError("resume-state raw_basis shape mismatch")
    if tuple(payload["coeff_weight"].shape) != tuple(coeff.weight.shape):
        raise ValueError("resume-state coeff shape mismatch")
    with torch.no_grad():
        raw_basis.copy_(payload["raw_basis"].to(device))
        coeff.weight.copy_(payload["coeff_weight"].to(device))
    basis_optimizer.load_state_dict(payload["basis_optimizer"])
    coeff_optimizer.load_state_dict(payload["coeff_optimizer"])
    basis_scheduler.load_state_dict(payload["basis_scheduler"])
    coeff_scheduler.load_state_dict(payload["coeff_scheduler"])
    generator.set_state(payload["generator_state"])
    order = payload["order"].long()
    cursor = int(payload["cursor"])
    sampling_weights = payload["sampling_weights"]
    if sampling_weights is not None:
        sampling_weights = sampling_weights.float()
    history = list(payload["history"])
    best = payload["best"]
    if best.get("basis") is not None:
        best["basis"] = best["basis"].cpu()
    if best.get("coeff") is not None:
        best["coeff"] = best["coeff"].cpu()
    start_step = int(payload["step"]) + 1
    print(json.dumps({
        "event": "full_state_loaded",
        "path": str(path),
        "resume_from_step": int(payload["step"]),
        "start_step": start_step,
    }), flush=True)
    return start_step, order, cursor, sampling_weights, history, best


@torch.no_grad()
def _evaluate_selected(
    lifted_train: Any,
    posenet: torch.nn.Module,
    masters: torch.Tensor,
    targets: torch.Tensor,
    coeff: torch.Tensor,
    basis: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    active_pair_ids: torch.Tensor,
) -> torch.Tensor:
    posenet.eval()
    values = []
    for start in range(0, len(active_pair_ids), args.eval_batch_size):
        end = min(start + args.eval_batch_size, len(active_pair_ids))
        batch_ids_cpu = active_pair_ids[start:end]
        batch_ids = batch_ids_cpu.to(device)
        if masters.device.type == device.type:
            master = masters.index_select(0, batch_ids).to(dtype=torch.float32)
        else:
            master = masters.index_select(0, batch_ids_cpu).to(device=device, dtype=torch.float32)
        pred = lifted_train.predict(
            posenet, master, coeff.index_select(0, batch_ids), basis,
            args.amplitude, args.carrier_base, args.master_carrier_amplitude,
        )
        values.append((pred - targets.index_select(0, batch_ids_cpu).to(device)).square().mean(1).cpu())
    return torch.cat(values)


def _active_pair_ids(args: argparse.Namespace) -> torch.Tensor:
    if args.smoke_pairs is None:
        return torch.arange(N_TOTAL_PAIRS, dtype=torch.long)
    if not 1 <= args.smoke_pairs <= 4:
        raise ValueError("--smoke-pairs is a smoke-only scope reduction and must be in [1,4]")
    return torch.linspace(0, N_TOTAL_PAIRS - 1, args.smoke_pairs).round().long()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--challenge-root", type=Path, required=True)
    parser.add_argument("--target-cache", type=Path, required=True)
    parser.add_argument("--master-checkpoint", type=Path, required=True)
    parser.add_argument("--init-carrier", type=Path, required=True)
    parser.add_argument("--master-cache", type=Path, default=None)
    parser.add_argument("--reuse-master-cache", action="store_true")
    parser.add_argument("--cache-masters-on-device", action="store_true")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--stop-after-step", type=int)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--eval-batch-size", type=int, default=12)
    parser.add_argument("--render-batch-size", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--state-save-every", type=int, default=250)
    parser.add_argument("--resume-state", type=Path)
    parser.add_argument("--lr-basis", type=float, default=0.003)
    parser.add_argument("--lr-coeff", type=float, default=0.03)
    parser.add_argument("--basis-freeze-fraction", type=float, default=0.30)
    parser.add_argument("--basis-train-until-fraction", type=float, default=1.0)
    parser.add_argument("--qat-fraction", type=float, default=0.65)
    parser.add_argument("--coeff-qat-fraction", type=float, default=None)
    parser.add_argument("--metric-loss-after-basis", action="store_true")
    parser.add_argument("--always-metric-loss", action="store_true")
    parser.add_argument("--metric-normalized-weight", type=float, default=0.0)
    parser.add_argument("--hard-mining-power", type=float, default=0.0)
    parser.add_argument("--hard-mining-max", type=float, default=8.0)
    parser.add_argument("--basis-bits", type=int, default=8)
    parser.add_argument("--coeff-bits", type=int, choices=(8, 10, 12, 16), default=8)
    parser.add_argument("--amplitude", type=float, default=32.0)
    parser.add_argument("--master-carrier-amplitude", type=float, default=0.0)
    parser.add_argument("--carrier-base", choices=("gray", "master"), default="gray")
    parser.add_argument("--zero-init-coeff", action="store_true")
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--row-local-mode",
        choices=(REFERENCE_SPARSE_MODE, DENSE_ADAPTER_MODE),
        default=REFERENCE_SPARSE_MODE,
        help=(
            "reference-sparse uses PR130's native sparse mechanism; "
            "dense-adapter is an explicit portability opt-in and never an "
            "automatic MPS fallback"
        ),
    )
    parser.add_argument("--smoke-pairs", type=int)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--save", type=Path, required=True)
    return parser.parse_args()


def run(
    args: argparse.Namespace, *, argv: list[str] | None = None
) -> dict[str, Any]:
    lifted_train = load_lifted_module("train_pose_carrier_full")
    if args.steps < 1:
        raise ValueError("--steps must be positive")
    stop_after = args.stop_after_step or args.steps
    if not 1 <= stop_after <= args.steps:
        raise ValueError("--stop-after-step must be in [1, --steps]")
    if args.state_save_every < 1:
        raise ValueError("--state-save-every must be positive")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    root = args.challenge_root.resolve()
    sys.path.insert(0, str(root))
    import modules  # pylint: disable=import-error,import-outside-toplevel

    active_pair_ids = _active_pair_ids(args)
    cache = torch.load(args.target_cache, map_location="cpu", weights_only=False)
    tokens = cache["seg"].long()
    targets = cache["pose"].float()
    if tokens.shape[0] != N_TOTAL_PAIRS or targets.shape != (N_TOTAL_PAIRS, 6):
        raise ValueError("target cache does not contain the expected 600 pairs")
    target_scale = (targets.amax(0) - targets.amin(0)).clamp_min(1e-4)

    master_checkpoint = torch.load(args.master_checkpoint, map_location="cpu", weights_only=False)
    master_config = architecture_config_from_checkpoint(
        master_checkpoint,
        consumer="train_pose_carrier_full_resumable.master",
    )
    master_model = lifted_train.SemanticTokenRenderer(
        width=int(master_config["width"]), blocks=int(master_config["blocks"]),
        frame_dim=int(master_config["frame_dim"]), num_pairs=N_TOTAL_PAIRS,
    ).eval().to(device)
    master_state = master_checkpoint["state_dict"]
    if "quant_bits" in master_checkpoint:
        from evaluate_semantic_quantization import quantize_tensor

        bits = int(master_checkpoint["quant_bits"])
        master_state = {
            name: quantize_tensor(
                value, bits, embedding=name.endswith("embed.weight")
            )[0]
            for name, value in master_state.items()
        }
    master_model.load_state_dict(master_state)
    masters = lifted_train.load_or_render_masters(args, master_model, tokens, device)
    del master_model
    clear_device_cache(device)
    if args.cache_masters_on_device:
        masters = masters.to(device=device, non_blocking=True)

    posenet = load_safetensors_cpu_then_move(
        modules.PoseNet().eval(), modules.posenet_sd_path, device
    )
    for parameter in posenet.parameters():
        parameter.requires_grad_(False)

    initial = torch.load(args.init_carrier, map_location="cpu", weights_only=False)
    raw_basis = torch.nn.Parameter(initial["basis"].float().to(device))
    basis_dim = raw_basis.shape[0]
    coeff, coeff_optimizer = build_row_local_coefficients(
        num_embeddings=N_TOTAL_PAIRS,
        embedding_dim=basis_dim,
        device=device,
        lr=args.lr_coeff,
        sparse_optimizer_type=lifted_train.RowLocalSparseAdam,
        mode=args.row_local_mode,
    )
    execution_provenance = _execution_provenance(
        args=args,
        coefficients=coeff,
        optimizer=coeff_optimizer,
        argv=list(sys.argv if argv is None else argv),
    )
    print(
        json.dumps(
            {
                "event": "row_local_optimizer_selected",
                **execution_provenance,
            }
        ),
        flush=True,
    )
    with torch.no_grad():
        coeff.weight.copy_(lifted_train.initialize_coefficients(
            initial, targets, target_scale, basis_dim, device
        ))
        if args.zero_init_coeff:
            coeff.weight.zero_()
    basis_optimizer = torch.optim.Adam([raw_basis], lr=args.lr_basis)
    basis_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        basis_optimizer, T_max=args.steps, eta_min=args.lr_basis * 0.01
    )
    coeff_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        coeff_optimizer, T_max=args.steps, eta_min=args.lr_coeff * 0.01
    )
    generator = torch.Generator().manual_seed(args.seed)
    order = torch.randperm(len(active_pair_ids), generator=generator)
    sampling_weights = None
    cursor = 0
    freeze_until = int(args.steps * args.basis_freeze_fraction)
    train_until = int(args.steps * args.basis_train_until_fraction)
    qat_start = int(args.steps * args.qat_fraction)
    coeff_qat_fraction = (
        args.qat_fraction if args.coeff_qat_fraction is None else args.coeff_qat_fraction
    )
    coeff_qat_start = int(args.steps * coeff_qat_fraction)
    history: list[dict[str, Any]] = []
    best: dict[str, Any] = {"mean": float("inf"), "basis": None, "coeff": None}
    start_step = 1

    if args.resume_state is not None:
        (
            start_step,
            order,
            cursor,
            sampling_weights,
            history,
            best,
        ) = _load_full_state(
            args.resume_state,
            args=args,
            raw_basis=raw_basis,
            coeff=coeff,
            basis_optimizer=basis_optimizer,
            coeff_optimizer=coeff_optimizer,
            basis_scheduler=basis_scheduler,
            coeff_scheduler=coeff_scheduler,
            generator=generator,
            active_pair_ids=active_pair_ids,
            device=device,
        )
    if start_step > stop_after:
        raise ValueError("resume-state is already beyond --stop-after-step")

    for step in range(start_step, stop_after + 1):
        if cursor + args.batch_size > len(active_pair_ids):
            if sampling_weights is None:
                order = torch.randperm(len(active_pair_ids), generator=generator)
            else:
                order = torch.multinomial(
                    sampling_weights,
                    len(active_pair_ids),
                    replacement=True,
                    generator=generator,
                )
            cursor = 0
        batch_pos = order[cursor:cursor + args.batch_size]
        cursor += args.batch_size
        batch_ids_cpu = active_pair_ids.index_select(0, batch_pos)
        batch_ids = batch_ids_cpu.to(device)
        if masters.device.type == device.type:
            master = masters.index_select(0, batch_ids).to(dtype=torch.float32)
        else:
            master = masters.index_select(0, batch_ids_cpu).to(device=device, dtype=torch.float32)
        target = targets.index_select(0, batch_ids_cpu).to(device)
        use_basis_qat = step > qat_start
        use_coeff_qat = step > coeff_qat_start
        train_basis = step > freeze_until and step <= train_until
        forward_basis = raw_basis if train_basis else raw_basis.detach()
        forward_coeff = coeff(batch_ids)
        if use_basis_qat:
            forward_basis = lifted_train.fake_quant_basis(forward_basis, args.basis_bits)
        if use_coeff_qat:
            forward_coeff = lifted_train.fake_quant_selected_coeff(
                forward_coeff, coeff.weight, args.coeff_bits
            )
        pred = lifted_train.predict(
            posenet, master, forward_coeff, forward_basis,
            args.amplitude, args.carrier_base, args.master_carrier_amplitude,
        )
        residual = pred - target
        normalized = residual / target_scale.to(device)
        use_metric_loss = args.always_metric_loss or (
            args.metric_loss_after_basis and step > train_until
        )
        if use_metric_loss:
            loss = (
                residual.square().mean()
                + args.metric_normalized_weight * normalized.square().mean()
            )
        else:
            loss = normalized.square().mean() + 0.02 * residual.square().mean()
        basis_optimizer.zero_grad(set_to_none=True)
        coeff_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([raw_basis], 10.0)
        prepare_row_local_step(coeff_optimizer, coeff.weight, batch_ids, 10.0)
        basis_optimizer.step()
        coeff_optimizer.step()
        basis_scheduler.step()
        coeff_scheduler.step()

        if step == 1 or step % 100 == 0:
            sample = residual.detach().square().mean(1)
            record = {
                "step": step,
                "phase": (
                    "full_qat" if use_basis_qat and use_coeff_qat
                    else "basis_qat" if use_basis_qat
                    else "float"
                ),
                "loss_mode": "raw_mse" if use_metric_loss else "range_normalized",
                "loss": float(loss.detach()),
                "sample": lifted_train.summarize(sample, 3e-5),
                "lr_basis": basis_optimizer.param_groups[0]["lr"],
                "lr_coeff": coeff_optimizer.param_groups[0]["lr"],
                "active_pair_ids": batch_ids_cpu.tolist(),
            }
            print(json.dumps(record), flush=True)

        should_eval = step % args.eval_every == 0 or step == stop_after
        should_state_save = step % args.state_save_every == 0 or step == stop_after
        if should_eval:
            basis_q, _, _ = lifted_train.quantize_basis(raw_basis.detach(), args.basis_bits)
            coeff_q, _, _ = lifted_train.quantize_coeff(coeff.weight.detach(), args.coeff_bits)
            eval_mse = _evaluate_selected(
                lifted_train, posenet, masters, targets, coeff_q, basis_q, args, device,
                active_pair_ids,
            )
            summary = lifted_train.summarize(eval_mse, 3e-5)
            record = {
                "step": step,
                "phase": "full_quantized",
                "scope_pairs": len(active_pair_ids),
                **summary,
            }
            history.append(record)
            print(json.dumps(record), flush=True)
            if args.hard_mining_power > 0.0:
                median = eval_mse.median().clamp_min(1e-12)
                relative = (eval_mse / median).clamp(1.0, args.hard_mining_max)
                sampling_weights = relative.pow(args.hard_mining_power)
                print(json.dumps({
                    "step": step,
                    "event": "update_hard_mining_weights",
                    "weight_min": float(sampling_weights.min()),
                    "weight_max": float(sampling_weights.max()),
                    "weight_mean": float(sampling_weights.mean()),
                    "scope_pairs": len(active_pair_ids),
                }), flush=True)
            latest_path = args.save.with_name(args.save.stem + ".latest.pt")
            checkpoint_payload = {
                "basis": raw_basis.detach().cpu().clone(),
                "coeff": coeff.weight.detach().cpu().clone(),
                "execution_provenance": execution_provenance,
                "result": {
                    "pair_ids": list(range(N_TOTAL_PAIRS)),
                    "step": step,
                    "quantized_basis_coeff": summary,
                    "per_pair_mse": eval_mse.tolist(),
                    "scope_pair_ids": active_pair_ids.tolist(),
                    "score_claim": False,
                    "execution_provenance": execution_provenance,
                },
            }
            _atomic_torch_save(checkpoint_payload, latest_path)
            if use_basis_qat and use_coeff_qat and summary["mean"] < best["mean"]:
                best = {
                    "mean": summary["mean"],
                    "basis": raw_basis.detach().cpu().clone(),
                    "coeff": coeff.weight.detach().cpu().clone(),
                }
                best_path = args.save.with_name(args.save.stem + ".best.pt")
                _atomic_torch_save(checkpoint_payload, best_path)
        if should_state_save:
            _save_full_state(
                args=args,
                step=step,
                raw_basis=raw_basis,
                coeff=coeff,
                basis_optimizer=basis_optimizer,
                coeff_optimizer=coeff_optimizer,
                basis_scheduler=basis_scheduler,
                coeff_scheduler=coeff_scheduler,
                generator=generator,
                order=order,
                cursor=cursor,
                sampling_weights=sampling_weights,
                history=history,
                best=best,
                active_pair_ids=active_pair_ids,
                execution_provenance=execution_provenance,
            )

    if best["basis"] is None:
        if not history:
            raise RuntimeError("no evaluation history was recorded")
        best = {
            "mean": history[-1]["mean"],
            "basis": raw_basis.detach().cpu().clone(),
            "coeff": coeff.weight.detach().cpu().clone(),
        }
    basis_best = best["basis"].to(device)
    coeff_best = best["coeff"].to(device)
    with torch.no_grad():
        fp_mse = _evaluate_selected(
            lifted_train, posenet, masters, targets, coeff_best, basis_best, args, device,
            active_pair_ids,
        )
        basis_q, basis_codes, basis_scales = lifted_train.quantize_basis(
            basis_best, args.basis_bits
        )
        coeff_q, coeff_codes, coeff_scales = lifted_train.quantize_coeff(
            coeff_best, args.coeff_bits
        )
        q_mse = _evaluate_selected(
            lifted_train, posenet, masters, targets, coeff_q, basis_q, args, device,
            active_pair_ids,
        )

    basis_values = int(raw_basis.numel())
    projected_bytes = (basis_values * args.basis_bits + 7) // 8 + 4 * basis_dim
    projected_bytes += (
        N_TOTAL_PAIRS * basis_dim * args.coeff_bits + 7
    ) // 8 + 4 * basis_dim
    q_summary = lifted_train.summarize(q_mse, 3e-5)
    full_scope = len(active_pair_ids) == N_TOTAL_PAIRS
    passed = q_summary["mean"] < 3e-6 and q_summary["reached"] == len(active_pair_ids)
    result = {
        "verdict": ("PASS" if passed else "FAIL") if full_scope else "SMOKE_ONLY",
        "schema": "ddm_mx2_resumable_pose_carrier_result.v1",
        "score_claim": False,
        "promotion_eligible": False,
        "config": _jsonable_args(args),
        "scope": {
            "full_scope": full_scope,
            "active_pair_ids": active_pair_ids.tolist(),
            "scope_reduction": (
                None if full_scope else "smoke-pairs validation only; not a family verdict"
            ),
        },
        "float": lifted_train.summarize(fp_mse, 3e-5),
        "quantized_basis_coeff": q_summary,
        "projected_600_payload_bytes": projected_bytes,
        "basis_code_range": [int(basis_codes.min()), int(basis_codes.max())],
        "basis_scale_range": [float(basis_scales.min()), float(basis_scales.max())],
        "coefficient_code_range": [int(coeff_codes.min()), int(coeff_codes.max())],
        "coefficient_scale_range": [float(coeff_scales.min()), float(coeff_scales.max())],
        "per_pair_mse": q_mse.tolist(),
        "history": history,
        "full_state_latest": str(_latest_state_path(args.save)),
        "execution_provenance": execution_provenance,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    _atomic_torch_save(
        {
            "basis": best["basis"],
            "coeff": best["coeff"],
            "execution_provenance": execution_provenance,
            "result": result,
        },
        args.save,
    )
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    args = parse_args()
    assert_governed_admission("pr130_pose_carrier_full_resumable")
    with lifted_script_path():
        run(args)


if __name__ == "__main__":
    main()
