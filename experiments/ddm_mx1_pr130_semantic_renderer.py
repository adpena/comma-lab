#!/usr/bin/env python
"""ddm_mx1 PR130 semantic renderer lift/port driver.

This is a scorer-slot-free harness for Row-1 readiness.  It can:

* run a tiny lifted-torch CPU smoke on real label tensors;
* probe local MLX availability without hiding runtime failures;
* run the torch-vs-MLX parity gate when MLX is available;
* emit a MAIN launch ticket for the n32 -> n120 stratified Metal run.

It does not run n600 scorer work and does not claim a contest score.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from tac.pr130_lift import SOURCE_REPO_HEAD, SOURCE_REPO_ROOT
from tac.pr130_lift.mlx_semantic_renderer import (
    MlxSemanticConfig,
    fake_quantize_parameter_tree,
    load_stage_checkpoint_npz,
    load_torch_state_dict_into_mlx,
    make_mlx_renderer,
    mlx_device_probe,
    require_mlx,
    save_stage_checkpoint_npz,
    curriculum_loss_mlx,
)

REPO = Path(__file__).resolve().parents[1]
LIFTED = REPO / "src" / "tac" / "pr130_lift" / "lifted"
SSD_ROOT = Path("/Volumes/VertigoDataTier/pact/ddm_mx1_20260806")
DEFAULT_INPUT_CACHE = Path("/Volumes/VertigoDataTier/pact/ddm_hb1_20260806/inputs/tq1c_seg_cache.pt")
DEFAULT_TARGET_CACHE = Path("/Volumes/VertigoDataTier/pact/ddm_hb1_20260806/inputs/gt_seg_cache.pt")
DEFAULT_INIT = Path(
    "/Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/"
    "repro_repo/artifacts/checkpoints/semantic_renderer_w96_b4_qat4_12k.pt"
)
CONTEST_DENOMINATOR_BYTES = 37_545_489


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_lifted_semantic() -> Any:
    if str(LIFTED) not in sys.path:
        sys.path.insert(0, str(LIFTED))
    spec = importlib.util.spec_from_file_location(
        "mx1_lifted_semantic_renderer_oracle",
        LIFTED / "semantic_renderer_oracle.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load lifted semantic_renderer_oracle.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FastTokenSegProxy(torch.nn.Module):
    """Fast differentiable proxy used only when upstream SegNet is not requested.

    The proxy consumes RGB and emits five logits.  It is deliberately labeled
    proxy and exists for checkpoint/resume smoke only; scorer-in-loop smoke uses
    upstream SegNet via ``--scorer upstream``.
    """

    def forward(self, rgb_nchw: torch.Tensor) -> torch.Tensor:
        x = rgb_nchw.float() / 255.0
        r, g, b = x[:, 0], x[:, 1], x[:, 2]
        return torch.stack(
            [
                2.0 * r - g - b,
                2.0 * g - r - b,
                2.0 * b - r - g,
                r + g - b,
                b + g - r,
            ],
            dim=1,
        )


def _load_upstream_segnet(device: torch.device) -> torch.nn.Module:
    root = REPO / "upstream"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import modules  # type: ignore[import-not-found]
    from safetensors.torch import load_file

    segnet = modules.SegNet().eval().to(device)
    segnet.load_state_dict(load_file(modules.segnet_sd_path, device=str(device)))
    for parameter in segnet.parameters():
        parameter.requires_grad_(False)
    return segnet


def _select_stratified_indices(n: int, total: int = 600, seed: int = 20260806) -> list[int]:
    rng = np.random.default_rng(seed)
    buckets = np.array_split(np.arange(total), n)
    selected = [int(rng.choice(bucket)) for bucket in buckets if len(bucket)]
    return sorted(selected)


def run_torch_smoke(args: argparse.Namespace) -> dict[str, Any]:
    lifted = _load_lifted_semantic()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cpu")
    target_cache = torch.load(args.target_cache, map_location="cpu", weights_only=False)
    target_tokens_all = target_cache["seg"].long()
    input_tokens_all = (
        torch.load(args.input_cache, map_location="cpu", weights_only=False)["seg"].long()
        if args.input_cache != args.target_cache
        else target_tokens_all
    )
    pair_ids = _select_stratified_indices(args.pairs, seed=args.seed)
    idx = torch.tensor(pair_ids, dtype=torch.long, device=device)
    conditioning = input_tokens_all[idx.cpu()].to(device)
    target = target_tokens_all[idx.cpu()].to(device)

    checkpoint = torch.load(args.init, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = lifted.SemanticTokenRenderer(
        width=int(config["width"]),
        blocks=int(config["blocks"]),
        frame_dim=int(config["frame_dim"]),
        num_pairs=600,
        phase_y=int(config.get("phase_y", 1)),
        phase_x=int(config.get("phase_x", 1)),
        temporal_radius=int(config.get("temporal_radius", 0)),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    segnet: torch.nn.Module
    scorer_axis: str
    if args.scorer == "upstream":
        segnet = _load_upstream_segnet(device)
        scorer_axis = "[macOS-CPU advisory torch upstream SegNet]"
    else:
        segnet = FastTokenSegProxy().eval().to(device)
        scorer_axis = "[proxy smoke no scorer authority]"
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.steps, 1), eta_min=args.lr * 0.01
    )
    history: list[dict[str, Any]] = []
    start_time = time.time()
    for step in range(args.steps):
        model.train()
        frame = lifted.render_for_seg(model, conditioning, idx, exact_path=args.train_exact_path)
        logits = segnet(frame)
        loss, phase = lifted.curriculum_loss(
            logits,
            target,
            step=step,
            total_steps=args.steps,
            ce_fraction=args.ce_fraction,
            softplus_fraction=args.softplus_fraction,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        scheduler.step()
        with torch.no_grad():
            pred = logits.argmax(dim=1)
            dseg = float((pred != target).float().mean())
        history.append(
            {
                "step": step + 1,
                "phase": phase,
                "loss": float(loss.detach()),
                "d_seg_batch": dseg,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )
    elapsed = time.time() - start_time
    stage_path = args.run_dir / f"torch_smoke_stage_steps{args.steps:04d}.pt"
    latest_path = args.run_dir / "torch_smoke.latest.pt"
    args.run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "config": config,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "history": history,
        "pair_ids": pair_ids,
        "seed": args.seed,
        "scorer_axis": scorer_axis,
        "score_claim": False,
    }
    torch.save(payload, stage_path)
    torch.save(payload, latest_path)
    resume = torch.load(latest_path, map_location="cpu", weights_only=False)
    resume_ok = sorted(resume.keys()) == sorted(payload.keys()) and resume["pair_ids"] == pair_ids
    return {
        "schema": "ddm_mx1_torch_smoke.v1",
        "status": "passed" if resume_ok else "blocked",
        "scorer_axis": scorer_axis,
        "score_claim": False,
        "pairs": pair_ids,
        "steps": args.steps,
        "elapsed_seconds": elapsed,
        "seconds_per_step": elapsed / max(args.steps, 1),
        "history": history,
        "checkpoint_stage": str(stage_path),
        "checkpoint_latest": str(latest_path),
        "checkpoint_latest_bytes": latest_path.stat().st_size,
        "checkpoint_latest_sha256": _sha256_file(latest_path),
        "resume_load_ok": resume_ok,
    }


def _build_torch_renderer(lifted: Any, checkpoint: Mapping[str, Any]) -> torch.nn.Module:
    config = checkpoint["config"]
    model = lifted.SemanticTokenRenderer(
        width=int(config["width"]),
        blocks=int(config["blocks"]),
        frame_dim=int(config["frame_dim"]),
        num_pairs=600,
        phase_y=int(config.get("phase_y", 1)),
        phase_x=int(config.get("phase_x", 1)),
        temporal_radius=int(config.get("temporal_radius", 0)),
    ).eval()
    model.load_state_dict(checkpoint["state_dict"])
    return model


def _as_float(value: Any) -> float:
    return float(np.asarray(value).reshape(()))


def run_mlx_parity(args: argparse.Namespace) -> dict[str, Any]:
    """Compare lifted torch CPU behavior to the MLX port on the selected host."""

    mx, _nn, _optim = require_mlx(device=args.device)
    from tac.local_acceleration.mlx_scorer_adapters import torch_segnet_to_mlx
    from tac.local_acceleration.pr95_hnerv_mlx_training import (
        apply_contest_faithful_roundtrip_nhwc,
    )

    lifted = _load_lifted_semantic()
    checkpoint = torch.load(args.init, map_location="cpu", weights_only=False)
    torch_model = _build_torch_renderer(lifted, checkpoint)
    config = MlxSemanticConfig.from_pr130_checkpoint_config(checkpoint["config"])
    mlx_model = make_mlx_renderer(config, device=args.device)
    load_torch_state_dict_into_mlx(
        mlx_model, checkpoint["state_dict"], device=args.device
    )

    input_tokens_all = torch.load(args.input_cache, map_location="cpu", weights_only=False)["seg"].long()
    target_tokens_all = torch.load(args.target_cache, map_location="cpu", weights_only=False)["seg"].long()
    pair_ids = _select_stratified_indices(args.pairs, seed=args.seed)
    idx_torch = torch.tensor(pair_ids, dtype=torch.long)
    conditioning_torch = input_tokens_all[pair_ids]
    target_torch = target_tokens_all[pair_ids]

    with torch.no_grad():
        torch_frame = torch_model(conditioning_torch, idx_torch)
        torch_frame_r = lifted.render_for_seg(
            torch_model, conditioning_torch, idx_torch, exact_path=True
        )
        segnet_torch = _load_upstream_segnet(torch.device("cpu"))
        torch_logits = segnet_torch(torch_frame_r)
        torch_loss, torch_phase = lifted.curriculum_loss(
            torch_logits,
            target_torch,
            step=0,
            total_steps=max(args.steps, 1),
            ce_fraction=args.ce_fraction,
            softplus_fraction=args.softplus_fraction,
        )

    conditioning_mlx = mx.array(conditioning_torch.numpy().astype(np.int32, copy=False))
    target_mlx = mx.array(target_torch.numpy().astype(np.int32, copy=False))
    idx_mlx = mx.array(np.asarray(pair_ids, dtype=np.int32))
    mlx_frame = mlx_model(conditioning_mlx, idx_mlx)
    mlx_frame_r = apply_contest_faithful_roundtrip_nhwc(
        mlx_frame, output_hw=(384, 512), ste_round=True
    )
    segnet_mlx = torch_segnet_to_mlx(segnet_torch)
    mlx_logits_nhwc = segnet_mlx(mlx_frame_r)
    mlx_logits_nchw = mx.transpose(mlx_logits_nhwc, (0, 3, 1, 2))
    mlx_loss, mlx_phase = curriculum_loss_mlx(
        mx,
        mlx_logits_nchw,
        target_mlx,
        step=0,
        total_steps=max(args.steps, 1),
        ce_fraction=args.ce_fraction,
        softplus_fraction=args.softplus_fraction,
    )
    mx.eval(mlx_frame, mlx_logits_nchw, mlx_loss)

    torch_frame_nhwc = torch_frame.detach().permute(0, 2, 3, 1).cpu().numpy()
    mlx_frame_np = np.asarray(mlx_frame)
    torch_pred = torch_logits.argmax(dim=1).cpu().numpy()
    mlx_pred = np.asarray(mx.argmax(mlx_logits_nchw, axis=1))
    frame_max_abs = float(np.max(np.abs(torch_frame_nhwc - mlx_frame_np)))
    argmax_diff_count = int(np.count_nonzero(torch_pred != mlx_pred))
    argmax_equal = argmax_diff_count == 0
    loss_abs = abs(float(torch_loss.detach()) - _as_float(mlx_loss))
    return {
        "schema": "ddm_mx1_mlx_parity.v1",
        "status": "passed",
        "axis": "[torch-CPU reference vs MLX host parity]",
        "score_claim": False,
        "parity_input": "real built label caches, not synthetic tensors",
        "token_batch_shape": list(conditioning_torch.shape),
        "scorer_batch_shape": list(torch_frame_r.shape),
        "scorer_adapter": "tac.local_acceleration.mlx_scorer_adapters.torch_segnet_to_mlx",
        "gradient_parity_claim": False,
        "gradient_parity_scope": "not measured by this mode; training telemetry remains research-signal unless a separate gradient-parity check is added",
        "pairs": pair_ids,
        "raw_frame_max_abs": frame_max_abs,
        "seg_argmax_equal": argmax_equal,
        "seg_argmax_diff_count": argmax_diff_count,
        "loss_abs_delta": loss_abs,
        "torch_phase": torch_phase,
        "mlx_phase": mlx_phase,
        "torch_loss": float(torch_loss.detach()),
        "mlx_loss": _as_float(mlx_loss),
    }


def run_mlx_train(args: argparse.Namespace) -> dict[str, Any]:
    """Run the real MLX Row-1 training path when MLX is available."""

    mx, nn, optim = require_mlx(device=args.device)
    from mlx.utils import tree_flatten, tree_unflatten

    from tac.local_acceleration.mlx_scorer_adapters import torch_segnet_to_mlx
    from tac.local_acceleration.pr95_hnerv_mlx_training import (
        apply_contest_faithful_roundtrip_nhwc,
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    pair_ids = _select_stratified_indices(args.pairs, seed=args.seed)
    checkpoint = torch.load(args.init, map_location="cpu", weights_only=False)
    config = MlxSemanticConfig.from_pr130_checkpoint_config(checkpoint["config"])
    config = MlxSemanticConfig(
        **(config.asdict() | {
            "bits": args.bits,
            "lr": args.lr,
            "steps": args.steps,
            "ce_fraction": args.ce_fraction,
            "softplus_fraction": args.softplus_fraction,
        })
    )
    model = make_mlx_renderer(config, device=args.device)
    optimizer = optim.AdamW(learning_rate=args.lr, weight_decay=0.0)
    start_step = 0
    history: list[dict[str, Any]] = []
    if args.resume_from is not None:
        resume = load_stage_checkpoint_npz(
            args.resume_from, model=model, optimizer=optimizer, mx=mx
        )
        start_step = int(resume["step"])
        history = list(resume["history"])
    else:
        load_torch_state_dict_into_mlx(
            model, checkpoint["state_dict"], device=args.device
        )

    input_tokens_all = torch.load(args.input_cache, map_location="cpu", weights_only=False)["seg"].long()
    target_tokens_all = torch.load(args.target_cache, map_location="cpu", weights_only=False)["seg"].long()
    conditioning_np = input_tokens_all[pair_ids].numpy().astype(np.int32, copy=False)
    target_np = target_tokens_all[pair_ids].numpy().astype(np.int32, copy=False)
    conditioning = mx.array(conditioning_np)
    target = mx.array(target_np)
    pair_idx = mx.array(np.asarray(pair_ids, dtype=np.int32))

    segnet_torch = _load_upstream_segnet(torch.device("cpu"))
    segnet_mlx = torch_segnet_to_mlx(segnet_torch)
    start_time = time.time()
    last_stage_path: Path | None = None

    def cosine_lr(step: int) -> float:
        progress = step / max(args.steps - 1, 1)
        return args.lr * (0.01 + 0.5 * (1.0 - 0.01) * (1.0 + math.cos(math.pi * progress)))

    for step in range(start_step, args.steps):
        optimizer.learning_rate = cosine_lr(step)
        base_params = model.trainable_parameters()

        def loss_for_params(params: Mapping[str, Any]) -> Any:
            if step < args.float_warmup_steps:
                active_params = params
                phase_prefix = "float_"
            else:
                active_params = fake_quantize_parameter_tree(
                    mx,
                    tree_flatten,
                    tree_unflatten,
                    params,
                    bits=args.bits,
                )
                phase_prefix = ""
            model.update(active_params)
            frame = model(conditioning, pair_idx)
            frame_r = apply_contest_faithful_roundtrip_nhwc(
                frame, output_hw=(384, 512), ste_round=True
            )
            logits_nhwc = segnet_mlx(frame_r)
            logits_nchw = mx.transpose(logits_nhwc, (0, 3, 1, 2))
            loss, phase = curriculum_loss_mlx(
                mx,
                logits_nchw,
                target,
                step=max(0, step - args.float_warmup_steps),
                total_steps=max(1, args.steps - args.float_warmup_steps),
                ce_fraction=args.ce_fraction,
                softplus_fraction=args.softplus_fraction,
            )
            loss_for_params.phase = phase_prefix + phase  # type: ignore[attr-defined]
            return loss

        value, grads = mx.value_and_grad(loss_for_params)(base_params)
        model.update(base_params)
        model.update(optimizer.apply_gradients(grads, base_params))
        mx.eval(model.parameters(), optimizer.state)
        phase = getattr(loss_for_params, "phase", "unknown")
        record: dict[str, Any] = {
            "step": step + 1,
            "phase": phase,
            "loss": float(value),
            "lr": float(optimizer.learning_rate),
        }
        if (step + 1) % max(args.eval_every, 1) == 0 or step + 1 == args.steps:
            frame = model(conditioning, pair_idx)
            frame_r = apply_contest_faithful_roundtrip_nhwc(
                frame, output_hw=(384, 512), ste_round=True
            )
            logits = segnet_mlx(frame_r)
            pred = mx.argmax(logits, axis=-1)
            dseg = mx.mean((pred != target).astype(mx.float32))
            mx.eval(dseg)
            record["d_seg_batch"] = float(dseg)
        history.append(record)
        if (step + 1) % max(args.checkpoint_every, 1) == 0 or step + 1 == args.steps:
            last_stage_path = args.run_dir / f"mlx_stage_step{step + 1:06d}.npz"
            save_stage_checkpoint_npz(
                last_stage_path,
                model=model,
                config=config,
                step=step + 1,
                history=history,
                optimizer_state=optimizer.state,
                extra={
                    "pair_ids": pair_ids,
                    "score_claim": False,
                    "axis": "[macOS-MLX research-signal]",
                    "source_repo_head": SOURCE_REPO_HEAD,
                },
            )
            latest = args.run_dir / "mlx.latest.npz"
            save_stage_checkpoint_npz(
                latest,
                model=model,
                config=config,
                step=step + 1,
                history=history,
                optimizer_state=optimizer.state,
                extra={
                    "pair_ids": pair_ids,
                    "score_claim": False,
                    "axis": "[macOS-MLX research-signal]",
                    "source_repo_head": SOURCE_REPO_HEAD,
                },
            )
    elapsed = time.time() - start_time
    latest_path = args.run_dir / "mlx.latest.npz"
    resume_check = load_stage_checkpoint_npz(latest_path, model=model, optimizer=optimizer, mx=mx)
    return {
        "schema": "ddm_mx1_mlx_train.v1",
        "status": "passed",
        "axis": "[macOS-MLX research-signal]",
        "score_claim": False,
        "pairs": pair_ids,
        "steps": args.steps,
        "start_step": start_step,
        "elapsed_seconds": elapsed,
        "seconds_per_step": elapsed / max(args.steps - start_step, 1),
        "history": history,
        "stage_checkpoint": str(last_stage_path) if last_stage_path else None,
        "latest_checkpoint": str(latest_path),
        "latest_checkpoint_bytes": latest_path.stat().st_size if latest_path.exists() else None,
        "latest_checkpoint_sha256": _sha256_file(latest_path) if latest_path.exists() else None,
        "resume_load": resume_check,
    }


def launch_ticket(args: argparse.Namespace, smoke: dict[str, Any] | None, mlx_probe: dict[str, Any]) -> dict[str, Any]:
    base_seconds = float(smoke["seconds_per_step"]) if smoke else None
    horizon = int(args.steps)
    if base_seconds is None:
        estimate = "blocked_local_mlx_probe_no_measured_mlx_step_time"
    else:
        estimate = {
            "local_torch_cpu_seconds_per_step": base_seconds,
            "naive_n32_seconds_at_same_backend": base_seconds * horizon * 32 / max(args.pairs, 1),
            "note": "MAIN must replace with first Metal n32 measured s/step; CPU smoke is not a Metal estimate.",
        }
    n32 = _select_stratified_indices(32, seed=args.seed)
    n120 = _select_stratified_indices(120, seed=args.seed + 1)
    return {
        "schema": "ddm_mx1_row1_launch_ticket.v1",
        "score_claim": False,
        "source_repo_root": SOURCE_REPO_ROOT,
        "source_repo_head": SOURCE_REPO_HEAD,
        "owned_run_root": str(args.run_dir),
        "input_cache": str(args.input_cache),
        "target_cache": str(args.target_cache),
        "init_checkpoint": str(args.init),
        "n32_stratified_indices": n32,
        "n120_stratified_indices": n120,
        "argv_n32": [
            ".venv/bin/python",
            "experiments/ddm_mx1_pr130_semantic_renderer.py",
            "--mode",
            "mlx-train",
            "--device",
            "gpu",
            "--pairs",
            "32",
            "--steps",
            str(horizon),
            "--lr",
            str(args.lr),
            "--ce-fraction",
            str(args.ce_fraction),
            "--softplus-fraction",
            str(args.softplus_fraction),
            "--bits",
            str(args.bits),
            "--seed",
            str(args.seed),
            "--checkpoint-every",
            str(args.checkpoint_every),
            "--eval-every",
            str(args.eval_every),
            "--input-cache",
            str(args.input_cache),
            "--target-cache",
            str(args.target_cache),
            "--init",
            str(args.init),
            "--run-dir",
            str(args.run_dir / "n32_metal"),
            "--out",
            str(args.run_dir / "n32_metal" / "result.json"),
        ],
        "argv_n120": [
            ".venv/bin/python",
            "experiments/ddm_mx1_pr130_semantic_renderer.py",
            "--mode",
            "mlx-train",
            "--device",
            "gpu",
            "--pairs",
            "120",
            "--steps",
            str(horizon),
            "--lr",
            str(args.lr),
            "--ce-fraction",
            str(args.ce_fraction),
            "--softplus-fraction",
            str(args.softplus_fraction),
            "--bits",
            str(args.bits),
            "--seed",
            str(args.seed + 1),
            "--checkpoint-every",
            str(args.checkpoint_every),
            "--eval-every",
            str(args.eval_every),
            "--input-cache",
            str(args.input_cache),
            "--target-cache",
            str(args.target_cache),
            "--init",
            str(args.init),
            "--run-dir",
            str(args.run_dir / "n120_metal"),
            "--out",
            str(args.run_dir / "n120_metal" / "result.json"),
        ],
        "verdict_protocol": {
            "axis": "[macOS-MLX research-signal] for train telemetry; frozen CPU-torch SegNet through exact R for d_seg; no contest promotion without upstream/evaluate.py on byte-closed archive",
            "compare_against": {
                "fp1_flat_paint_floor_d_seg": 0.008305,
                "pr130_external_d_seg": 0.00029660,
            },
            "selection": "stratified-random n32, then n120; never prefix; n600 only after scorer slot assignment",
        },
        "memory_projection": {
            "renderer_width": 96,
            "blocks": 4,
            "stage": "PR130 stage 08 tail from retained semantic_renderer_w96_b4_qat4_12k checkpoint",
            "configured_horizon_steps": horizon,
            "lr": args.lr,
            "ce_fraction": args.ce_fraction,
            "softplus_fraction": args.softplus_fraction,
            "checkpoint_size_source_bytes": args.init.stat().st_size if args.init.exists() else None,
            "label_cache_inputs_bytes": args.input_cache.stat().st_size if args.input_cache.exists() else None,
            "label_cache_targets_bytes": args.target_cache.stat().st_size if args.target_cache.exists() else None,
        },
        "wall_clock_estimate": estimate,
        "local_mlx_probe": mlx_probe,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["probe", "torch-smoke", "mlx-parity", "mlx-train"], default="probe")
    parser.add_argument("--input-cache", type=Path, default=DEFAULT_INPUT_CACHE)
    parser.add_argument("--target-cache", type=Path, default=DEFAULT_TARGET_CACHE)
    parser.add_argument("--init", type=Path, default=DEFAULT_INIT)
    parser.add_argument("--run-dir", type=Path, default=SSD_ROOT)
    parser.add_argument("--pairs", type=int, default=2)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--lr", type=float, default=2e-7)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--ce-fraction", type=float, default=0.0)
    parser.add_argument("--softplus-fraction", type=float, default=-999.0)
    parser.add_argument("--train-exact-path", action="store_true")
    parser.add_argument("--scorer", choices=["upstream", "proxy"], default="upstream")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--float-warmup-steps", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--out", type=Path, default=SSD_ROOT / "mx1_driver_result.json")
    args = parser.parse_args()

    result: dict[str, Any] = {
        "schema": "ddm_mx1_pr130_semantic_renderer_driver.v1",
        "mode": args.mode,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "score_claim": False,
        "source_repo_head": SOURCE_REPO_HEAD,
        "source_repo_root": SOURCE_REPO_ROOT,
    }
    mlx_probe = mlx_device_probe(device=args.device)
    result["mlx_probe"] = mlx_probe
    smoke: dict[str, Any] | None = None
    if args.mode == "torch-smoke":
        smoke = run_torch_smoke(args)
        result["torch_smoke"] = smoke
    elif args.mode == "mlx-parity":
        if mlx_probe["status"] == "blocked":
            result["status"] = "blocked"
            result["blocker"] = "local MLX runtime unavailable; run parity on MAIN Metal host"
        else:
            result["mlx_parity"] = run_mlx_parity(args)
            result["status"] = result["mlx_parity"]["status"]
    elif args.mode == "mlx-train":
        if mlx_probe["status"] == "blocked":
            result["status"] = "blocked"
            result["blocker"] = "local MLX runtime unavailable; run the launch-ticket argv on MAIN Metal host"
        else:
            result["mlx_train"] = run_mlx_train(args)
            result["status"] = result["mlx_train"]["status"]
    result["launch_ticket"] = launch_ticket(args, smoke, mlx_probe)
    write_json(args.out, result)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    if args.mode in {"mlx-parity", "mlx-train"} and mlx_probe["status"] == "blocked":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
