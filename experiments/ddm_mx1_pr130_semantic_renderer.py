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
import gc
import hashlib
import importlib.util
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tac.pr130_lift import SOURCE_REPO_HEAD, SOURCE_REPO_ROOT
from tac.pr130_lift.mlx_semantic_renderer import (
    MlxSemanticConfig,
    MlxUnavailableError,
    curriculum_loss_mlx,
    fake_quantize_parameter_tree,
    load_stage_checkpoint_npz,
    load_torch_state_dict_into_mlx,
    make_mlx_renderer,
    mlx_device_probe,
    require_mlx,
    save_stage_checkpoint_npz,
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
GIB = 1024.0 ** 3
MEM_PROBE_RECEIPT_SCHEMA = "ddm_mx1_load_phase_peak_receipt.v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _gib_or_none(num_bytes: int | float | None) -> float | None:
    if num_bytes is None:
        return None
    return round(float(num_bytes) / GIB, 6)


def _process_rss_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:
        pass
    try:
        out = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(os.getpid())],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            return int(out.stdout.strip().split()[0]) * 1024
    except Exception:
        return None
    return None


def _system_available_bytes() -> int | None:
    try:
        import psutil

        # RAW_VM_BASIS_OK: telemetry-only load-phase receipt/default limit hint,
        # not an admission guard or launch clearance.
        return int(psutil.virtual_memory().available)
    except Exception:
        return None


def _mlx_allocator_bytes(mx: Any | None) -> dict[str, int | None]:
    if mx is None:
        return {"active": None, "cache": None, "peak": None}
    out: dict[str, int | None] = {}
    for key, names in {
        "active": ("get_active_memory", "metal.get_active_memory"),
        "cache": ("get_cache_memory", "metal.get_cache_memory"),
        "peak": ("get_peak_memory", "metal.get_peak_memory"),
    }.items():
        value: int | None = None
        for name in names:
            try:
                obj: Any = mx
                for part in name.split("."):
                    obj = getattr(obj, part)
                value = int(obj())
                break
            except Exception:
                continue
        out[key] = value
    return out


class LoadPhaseMemoryProbe:
    """Small typed recorder for load-phase RSS + MLX allocator telemetry."""

    def __init__(self) -> None:
        self.samples: list[dict[str, Any]] = []
        self._start_rss_bytes: int | None = None
        self._start_available_bytes: int | None = None

    def sample(self, stage: str, *, mx: Any | None = None, note: str | None = None) -> dict[str, Any]:
        rss_bytes = _process_rss_bytes()
        available_bytes = _system_available_bytes()
        if self._start_rss_bytes is None:
            self._start_rss_bytes = rss_bytes
        if self._start_available_bytes is None:
            self._start_available_bytes = available_bytes
        mlx_bytes = _mlx_allocator_bytes(mx)
        sample = {
            "event_index": len(self.samples),
            "stage": stage,
            "timestamp_utc": _utc_now_iso(),
            "rss_gib": _gib_or_none(rss_bytes),
            "rss_delta_from_start_gib": _gib_or_none(
                None if rss_bytes is None or self._start_rss_bytes is None else rss_bytes - self._start_rss_bytes
            ),
            "sys_available_gib": _gib_or_none(available_bytes),
            "sys_available_delta_from_start_gib": _gib_or_none(
                None
                if available_bytes is None or self._start_available_bytes is None
                else available_bytes - self._start_available_bytes
            ),
            "mlx_active_gib": _gib_or_none(mlx_bytes["active"]),
            "mlx_cache_gib": _gib_or_none(mlx_bytes["cache"]),
            "mlx_peak_gib": _gib_or_none(mlx_bytes["peak"]),
        }
        if note:
            sample["note"] = note
        self.samples.append(sample)
        return sample

    def peak(self) -> dict[str, Any]:
        def max_present(key: str) -> float | None:
            values = [row[key] for row in self.samples if row.get(key) is not None]
            return max(values) if values else None

        def min_present(key: str) -> float | None:
            values = [row[key] for row in self.samples if row.get(key) is not None]
            return min(values) if values else None

        return {
            "sample_count": len(self.samples),
            "peak_rss_gib": max_present("rss_gib"),
            "min_sys_available_gib": min_present("sys_available_gib"),
            "peak_mlx_active_gib": max_present("mlx_active_gib"),
            "peak_mlx_cache_gib": max_present("mlx_cache_gib"),
            "peak_mlx_reported_gib": max_present("mlx_peak_gib"),
        }


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


def _load_selected_seg_tokens(cache_path: Path, pair_ids: list[int]) -> tuple[torch.Tensor, dict[str, Any]]:
    """Load only selected cache rows into the retained tensor.

    The cache file is still a monolithic ``torch.save`` payload, so PyTorch must
    deserialize it. The fix is to index+clone immediately and drop the full
    cache before any MLX arrays or scorer weights are built.
    """

    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    try:
        seg_all = payload["seg"]
        idx = torch.tensor(pair_ids, dtype=torch.long)
        selected = seg_all.index_select(0, idx).long().clone().contiguous()
        meta = {
            "cache_path": str(cache_path),
            "cache_bytes": cache_path.stat().st_size if cache_path.exists() else None,
            "full_shape_seen": list(seg_all.shape),
            "full_dtype_seen": str(seg_all.dtype),
            "selected_shape": list(selected.shape),
            "selected_dtype": str(selected.dtype),
            "selected_pair_count": len(pair_ids),
        }
    finally:
        del payload
        if "seg_all" in locals():
            del seg_all
        gc.collect()
    return selected, meta


def _load_selected_token_arrays(
    *,
    input_cache: Path,
    target_cache: Path,
    pair_ids: list[int],
    memory_probe: LoadPhaseMemoryProbe | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if memory_probe is not None:
        memory_probe.sample("before_selected_cache_load")
    input_tokens, input_meta = _load_selected_seg_tokens(input_cache, pair_ids)
    if memory_probe is not None:
        memory_probe.sample("after_input_cache_selected_clone")
    if input_cache == target_cache:
        target_tokens = input_tokens
        target_meta = {**input_meta, "shared_with_input_cache": True}
    else:
        target_tokens, target_meta = _load_selected_seg_tokens(target_cache, pair_ids)
        if memory_probe is not None:
            memory_probe.sample("after_target_cache_selected_clone")

    conditioning_np = input_tokens.numpy().astype(np.int32, copy=True)
    target_np = target_tokens.numpy().astype(np.int32, copy=True)
    del input_tokens
    del target_tokens
    gc.collect()
    if memory_probe is not None:
        memory_probe.sample("after_selected_cache_numpy_copy_and_torch_free")
    return conditioning_np, target_np, {
        "input": input_meta,
        "target": target_meta,
        "subset_before_materialize": "torch.load_index_clone_del_full_cache",
    }


def run_torch_smoke(args: argparse.Namespace) -> dict[str, Any]:
    lifted = _load_lifted_semantic()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cpu")
    pair_ids = _select_stratified_indices(args.pairs, seed=args.seed)
    conditioning_np, target_np, cache_meta = _load_selected_token_arrays(
        input_cache=args.input_cache,
        target_cache=args.target_cache,
        pair_ids=pair_ids,
        memory_probe=None,
    )
    idx = torch.tensor(pair_ids, dtype=torch.long, device=device)
    conditioning = torch.from_numpy(conditioning_np).long().to(device)
    target = torch.from_numpy(target_np).long().to(device)
    del conditioning_np, target_np
    gc.collect()

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
        "cache_load": cache_meta,
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

    pair_ids = _select_stratified_indices(args.pairs, seed=args.seed)
    idx_torch = torch.tensor(pair_ids, dtype=torch.long)
    conditioning_np, target_np, cache_meta = _load_selected_token_arrays(
        input_cache=args.input_cache,
        target_cache=args.target_cache,
        pair_ids=pair_ids,
        memory_probe=None,
    )
    conditioning_torch = torch.from_numpy(conditioning_np).long()
    target_torch = torch.from_numpy(target_np).long()
    del conditioning_np, target_np
    gc.collect()

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
        "cache_load": cache_meta,
        "raw_frame_max_abs": frame_max_abs,
        "seg_argmax_equal": argmax_equal,
        "seg_argmax_diff_count": argmax_diff_count,
        "loss_abs_delta": loss_abs,
        "torch_phase": torch_phase,
        "mlx_phase": mlx_phase,
        "torch_loss": float(torch_loss.detach()),
        "mlx_loss": _as_float(mlx_loss),
    }


def _derive_mem_budget_gb(explicit_gb: float | None) -> dict[str, Any]:
    if explicit_gb is not None:
        if explicit_gb <= 0:
            raise ValueError("--mem-budget-gb must be positive when provided")
        return {
            "budget_gb": float(explicit_gb),
            "source": "explicit_cli",
            "available_gib_at_start": _gib_or_none(_system_available_bytes()),
        }
    available = _system_available_bytes()
    if available is None:
        return {
            "budget_gb": None,
            "source": "unavailable_no_limit_applied",
            "available_gib_at_start": None,
        }
    available_gib = float(available) / GIB
    return {
        "budget_gb": round(max(1.0, available_gib * 0.50), 3),
        "source": "default_50pct_of_available_memory_at_start",
        "available_gib_at_start": round(available_gib, 6),
    }


def _call_optional_mlx_limit(mx: Any, dotted_name: str, value: int) -> dict[str, Any]:
    try:
        obj: Any = mx
        for part in dotted_name.split("."):
            obj = getattr(obj, part)
        obj(value)
        return {"target": dotted_name, "status": "applied", "value_bytes": value}
    except AttributeError:
        return {"target": dotted_name, "status": "unavailable", "value_bytes": value}
    except Exception as exc:
        return {
            "target": dotted_name,
            "status": "failed",
            "value_bytes": value,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def _configure_mlx_memory_limits(mx: Any, explicit_gb: float | None) -> dict[str, Any]:
    derived = _derive_mem_budget_gb(explicit_gb)
    budget_gb = derived["budget_gb"]
    if budget_gb is None:
        return {**derived, "memory_limit": None, "cache_limit": None, "calls": []}
    memory_limit = int(float(budget_gb) * GIB)
    cache_limit = int(max(256 * 1024 * 1024, memory_limit * 0.25))
    calls = [
        _call_optional_mlx_limit(mx, "set_memory_limit", memory_limit),
        _call_optional_mlx_limit(mx, "metal.set_memory_limit", memory_limit),
        _call_optional_mlx_limit(mx, "set_cache_limit", cache_limit),
        _call_optional_mlx_limit(mx, "metal.set_cache_limit", cache_limit),
    ]
    return {
        **derived,
        "memory_limit": memory_limit,
        "cache_limit": cache_limit,
        "calls": calls,
    }


def _clear_mlx_cache(mx: Any) -> None:
    for name in ("clear_cache", "metal.clear_cache"):
        try:
            obj: Any = mx
            for part in name.split("."):
                obj = getattr(obj, part)
            obj()
            return
        except Exception:
            continue


def _mx_eval_setup_barrier(
    mx: Any,
    memory_probe: LoadPhaseMemoryProbe,
    stage: str,
    *values: Any,
    note: str | None = None,
) -> None:
    if values:
        mx.eval(*values)
    memory_probe.sample(stage, mx=mx, note=note)


def run_mlx_train(
    args: argparse.Namespace,
    *,
    memory_probe: LoadPhaseMemoryProbe | None = None,
) -> dict[str, Any]:
    """Run the real MLX Row-1 training path when MLX is available."""

    probe = memory_probe if memory_probe is not None else LoadPhaseMemoryProbe()
    probe.sample("start")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    pair_ids = _select_stratified_indices(args.pairs, seed=args.seed)
    checkpoint = torch.load(args.init, map_location="cpu", weights_only=False)
    probe.sample("after_init_checkpoint_torch_load")
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
    conditioning_np, target_np, cache_meta = _load_selected_token_arrays(
        input_cache=args.input_cache,
        target_cache=args.target_cache,
        pair_ids=pair_ids,
        memory_probe=probe,
    )

    mx, _nn, optim = require_mlx(device=args.device)
    memory_limits = _configure_mlx_memory_limits(mx, args.mem_budget_gb)
    probe.sample("after_require_mlx_and_memory_limits", mx=mx)
    from mlx.utils import tree_flatten, tree_unflatten

    from tac.local_acceleration.mlx_scorer_adapters import torch_segnet_to_mlx
    from tac.local_acceleration.pr95_hnerv_mlx_training import (
        apply_contest_faithful_roundtrip_nhwc,
    )

    model = make_mlx_renderer(config, device=args.device)
    optimizer = optim.AdamW(learning_rate=args.lr, weight_decay=0.0)
    _mx_eval_setup_barrier(mx, probe, "after_model_init", model.parameters())
    start_step = 0
    history: list[dict[str, Any]] = []
    if args.resume_from is not None:
        resume = load_stage_checkpoint_npz(
            args.resume_from, model=model, optimizer=optimizer, mx=mx
        )
        start_step = int(resume["step"])
        history = list(resume["history"])
        del checkpoint
        gc.collect()
        _clear_mlx_cache(mx)
        probe.sample("after_resume_load_and_init_checkpoint_free", mx=mx)
    else:
        state_dict = checkpoint["state_dict"]
        load_torch_state_dict_into_mlx(
            model, state_dict, device=args.device
        )
        _mx_eval_setup_barrier(mx, probe, "after_model_weight_mlx_conversion", model.parameters())
        del state_dict
        del checkpoint
        gc.collect()
        _clear_mlx_cache(mx)
        probe.sample("after_torch_checkpoint_free", mx=mx)

    conditioning = mx.array(conditioning_np)
    target = mx.array(target_np)
    pair_idx = mx.array(np.asarray(pair_ids, dtype=np.int32))
    _mx_eval_setup_barrier(
        mx,
        probe,
        "after_selected_tokens_mlx_conversion",
        conditioning,
        target,
        pair_idx,
    )
    del conditioning_np, target_np
    gc.collect()
    _clear_mlx_cache(mx)
    probe.sample("after_selected_token_numpy_free", mx=mx)

    segnet_torch = _load_upstream_segnet(torch.device("cpu"))
    probe.sample("after_upstream_segnet_torch_load", mx=mx)
    segnet_mlx = torch_segnet_to_mlx(segnet_torch)
    segnet_params = segnet_mlx.parameters() if hasattr(segnet_mlx, "parameters") else []
    _mx_eval_setup_barrier(mx, probe, "after_segnet_mlx_conversion", segnet_params)
    del segnet_torch
    gc.collect()
    _clear_mlx_cache(mx)
    probe.sample("after_segnet_torch_free", mx=mx)
    start_time = time.time()
    last_stage_path: Path | None = None

    def cosine_lr(step: int) -> float:
        progress = step / max(args.steps - 1, 1)
        return args.lr * (0.01 + 0.5 * (1.0 - 0.01) * (1.0 + math.cos(math.pi * progress)))

    for step in range(start_step, args.steps):
        optimizer.learning_rate = cosine_lr(step)
        base_params = model.trainable_parameters()
        step_for_loss = step

        def loss_for_params(params: Mapping[str, Any], *, step_for_loss: int = step_for_loss) -> Any:
            if step_for_loss < args.float_warmup_steps:
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
                step=max(0, step_for_loss - args.float_warmup_steps),
                total_steps=max(1, args.steps - args.float_warmup_steps),
                ce_fraction=args.ce_fraction,
                softplus_fraction=args.softplus_fraction,
            )
            loss_for_params.phase = phase_prefix + phase  # type: ignore[attr-defined]
            return loss

        value, grads = mx.value_and_grad(loss_for_params)(base_params)
        model.update(base_params)
        model.update(optimizer.apply_gradients(grads, base_params))
        mx.eval(value, model.parameters(), optimizer.state)
        phase = getattr(loss_for_params, "phase", "unknown")
        record: dict[str, Any] = {
            "step": step + 1,
            "phase": phase,
            "loss": float(value),
            "lr": float(optimizer.learning_rate),
        }
        if step == start_step or args.steps <= 3:
            probe.sample(f"after_train_step_{step + 1:06d}", mx=mx)
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
            if args.steps <= 3:
                probe.sample(f"after_eval_step_{step + 1:06d}", mx=mx)
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
            if args.steps <= 3:
                probe.sample(f"after_checkpoint_step_{step + 1:06d}", mx=mx)
    elapsed = time.time() - start_time
    latest_path = args.run_dir / "mlx.latest.npz"
    resume_check = load_stage_checkpoint_npz(latest_path, model=model, optimizer=optimizer, mx=mx)
    probe.sample("after_resume_check", mx=mx)
    return {
        "schema": "ddm_mx1_mlx_train.v1",
        "status": "passed",
        "axis": "[macOS-MLX research-signal]",
        "score_claim": False,
        "pairs": pair_ids,
        "cache_load": cache_meta,
        "memory_limits": memory_limits,
        "load_memory_samples": probe.samples,
        "load_memory_peak": probe.peak(),
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


def _mem_probe_args(args: argparse.Namespace) -> argparse.Namespace:
    payload = vars(args).copy()
    payload["steps"] = int(args.mem_probe_steps)
    payload["eval_every"] = 1
    payload["checkpoint_every"] = max(1, int(args.mem_probe_steps))
    return argparse.Namespace(**payload)


def _build_mem_probe_receipt(
    *,
    args: argparse.Namespace,
    probe_args: argparse.Namespace,
    probe: LoadPhaseMemoryProbe,
    status: str,
    train_result: dict[str, Any] | None,
    blocker: dict[str, Any] | None,
) -> dict[str, Any]:
    required_stage = f"after_train_step_{int(probe_args.steps):06d}"
    final_step_sample = next(
        (row for row in probe.samples if row.get("stage") == required_stage),
        None,
    )
    has_mlx_allocator_telemetry = bool(
        final_step_sample
        and any(
            final_step_sample.get(key) is not None
            for key in ("mlx_active_gib", "mlx_cache_gib", "mlx_peak_gib")
        )
    )
    metal_fire_clearance = status == "passed" and final_step_sample is not None and has_mlx_allocator_telemetry
    return {
        "schema": MEM_PROBE_RECEIPT_SCHEMA,
        "status": status,
        "axis": "[load-phase memory telemetry; score_claim=false]",
        "score_claim": False,
        "source_repo_head": SOURCE_REPO_HEAD,
        "source_repo_root": SOURCE_REPO_ROOT,
        "mode": "mem-probe",
        "device_request": args.device,
        "pairs": int(args.pairs),
        "pair_ids": _select_stratified_indices(args.pairs, seed=args.seed),
        "requested_training_steps": int(probe_args.steps),
        "mem_budget_gb_arg": args.mem_budget_gb,
        "input_cache": str(args.input_cache),
        "target_cache": str(args.target_cache),
        "init_checkpoint": str(args.init),
        "samples": probe.samples,
        "peak": probe.peak(),
        "train_result_summary": None
        if train_result is None
        else {
            "status": train_result.get("status"),
            "steps": train_result.get("steps"),
            "seconds_per_step": train_result.get("seconds_per_step"),
            "stage_checkpoint": train_result.get("stage_checkpoint"),
            "latest_checkpoint": train_result.get("latest_checkpoint"),
            "latest_checkpoint_sha256": train_result.get("latest_checkpoint_sha256"),
            "load_memory_peak": train_result.get("load_memory_peak"),
        },
        "blocker": blocker,
        "clearance_checks": {
            "required_stage": required_stage,
            "has_required_stage_sample": final_step_sample is not None,
            "has_mlx_allocator_telemetry_at_required_stage": has_mlx_allocator_telemetry,
        },
        "metal_fire_clearance": metal_fire_clearance,
        "clearance_rule": (
            "A Metal launch may consume this receipt only when status=passed, "
            "samples include the required final mem-probe train step with MLX allocator telemetry, and peak fits the composed "
            "one-Metal-fire-at-a-time schedule under the host ceiling."
        ),
    }


def run_mem_probe(args: argparse.Namespace) -> dict[str, Any]:
    probe = LoadPhaseMemoryProbe()
    probe_args = _mem_probe_args(args)
    train_result: dict[str, Any] | None = None
    blocker: dict[str, Any] | None = None
    status = "passed"
    try:
        train_result = run_mlx_train(probe_args, memory_probe=probe)
        status = str(train_result.get("status", "passed"))
    except Exception as exc:
        status = "blocked"
        blocker = {
            "error_type": type(exc).__name__,
            "error": str(exc),
            "boundary": (
                "CPU load-path telemetry may be present before this blocker; "
                "full MLX allocator/three-step telemetry requires MAIN Metal."
            ),
        }
        if isinstance(exc, MlxUnavailableError):
            blocker["verdict_scope"] = "ENVIRONMENT: local sandbox MLX/Metal unavailable"
        else:
            blocker["verdict_scope"] = "INSTANCE: mem-probe execution"
    receipt = _build_mem_probe_receipt(
        args=args,
        probe_args=probe_args,
        probe=probe,
        status=status,
        train_result=train_result,
        blocker=blocker,
    )
    receipt_path = args.run_dir / "mem_probe_receipt.json"
    write_json(receipt_path, receipt)
    return {
        "schema": "ddm_mx1_mem_probe.v1",
        "status": status,
        "score_claim": False,
        "receipt_path": str(receipt_path),
        "receipt_sha256": _sha256_file(receipt_path),
        "receipt": receipt,
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

    # RR3-F1: a Row-1 verdict requires TWO arms — ARM-CAP (GT tokens -> GT targets, receiver
    # CAPACITY vs the fp1 flat-paint floor and PR130's external number) and ARM-VEH (public-wire
    # tq1c tokens -> GT targets, composed-vehicle correction reach). A single-arm ticket
    # conflates the two questions, so the bare argv_n32/argv_n120 keys no longer exist.
    def _arm_argv(pairs: int, seed: int, input_cache: Path, subdir: str) -> list[str]:
        run_dir = args.run_dir / subdir
        argv = [
            ".venv/bin/python",
            "experiments/ddm_mx1_pr130_semantic_renderer.py",
            "--mode", "mlx-train",
            "--device", "gpu",
            "--pairs", str(pairs),
            "--steps", str(horizon),
            "--lr", str(args.lr),
            "--ce-fraction", str(args.ce_fraction),
            "--softplus-fraction", str(args.softplus_fraction),
            "--bits", str(args.bits),
            "--seed", str(seed),
            "--checkpoint-every", str(args.checkpoint_every),
            "--eval-every", str(args.eval_every),
            "--input-cache", str(input_cache),
            "--target-cache", str(args.target_cache),
            "--init", str(args.init),
            "--run-dir", str(run_dir),
            "--out", str(run_dir / "result.json"),
        ]
        if args.mem_budget_gb is not None:
            argv.extend(["--mem-budget-gb", str(args.mem_budget_gb)])
        return argv

    mem_probe_receipt_path = args.run_dir / "mem_probe_receipt.json"
    mem_probe_command = [
        ".venv/bin/python",
        "experiments/ddm_mx1_pr130_semantic_renderer.py",
        "--mode", "mem-probe",
        "--device", "gpu",
        "--pairs", "32",
        "--mem-probe-steps", str(args.mem_probe_steps),
        "--lr", str(args.lr),
        "--ce-fraction", str(args.ce_fraction),
        "--softplus-fraction", str(args.softplus_fraction),
        "--bits", str(args.bits),
        "--seed", str(args.seed),
        "--checkpoint-every", str(max(1, int(args.mem_probe_steps))),
        "--eval-every", "1",
        "--input-cache", str(args.input_cache),
        "--target-cache", str(args.target_cache),
        "--init", str(args.init),
        "--run-dir", str(args.run_dir),
        "--out", str(args.run_dir / "mem_probe_result.json"),
    ]
    if args.mem_budget_gb is not None:
        mem_probe_command.extend(["--mem-budget-gb", str(args.mem_budget_gb)])

    cap_cache = args.target_cache  # GT labels as tokens AND targets
    veh_cache = args.input_cache   # public-wire (tq1c) labels as tokens, GT targets
    return {
        "schema": "ddm_mx1_row1_launch_ticket.v2_two_arm",
        "score_claim": False,
        "mem_probe_receipt_required": True,
        "mem_probe_receipt_path": str(mem_probe_receipt_path),
        "mem_probe_command": mem_probe_command,
        "scheduling": (
            "SEQUENTIAL — one Metal arm at a time (operator machine OOM 2026-08-06); "
            "ARM-VEH fires only after ARM-CAP completes or a composed measured-peak "
            "projection shows headroom under 116GiB"
        ),
        "source_repo_root": SOURCE_REPO_ROOT,
        "source_repo_head": SOURCE_REPO_HEAD,
        "owned_run_root": str(args.run_dir),
        "input_cache": str(args.input_cache),
        "target_cache": str(args.target_cache),
        "init_checkpoint": str(args.init),
        "n32_stratified_indices": n32,
        "n120_stratified_indices": n120,
        "arm_selection_rule": (
            "fire BOTH n32 arms (arm_cap: GT->GT receiver capacity; arm_veh: tq1c->GT composed-"
            "vehicle reach); NO n120 dispatch until the scaled arm is explicitly selected from "
            "the two n32 CPU-torch verdicts; MLX telemetry is research-signal only"
        ),
        "argv_n32_arm_cap": _arm_argv(32, args.seed, cap_cache, "launch_arm_cap/n32_metal"),
        "argv_n32_arm_veh": _arm_argv(32, args.seed, veh_cache, "launch_arm_veh/n32_metal"),
        "argv_n120_arm_cap": _arm_argv(120, args.seed + 1, cap_cache, "launch_arm_cap/n120_metal"),
        "argv_n120_arm_veh": _arm_argv(120, args.seed + 1, veh_cache, "launch_arm_veh/n120_metal"),
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
            "mem_budget_gb_arg": args.mem_budget_gb,
            "mem_budget_default_policy": "50% of available memory at process start when --mem-budget-gb is omitted",
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
    parser.add_argument(
        "--mode",
        choices=["probe", "torch-smoke", "mlx-parity", "mlx-train", "mem-probe"],
        default="probe",
    )
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
    parser.add_argument("--mem-budget-gb", type=float)
    parser.add_argument("--mem-probe-steps", type=int, default=3)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--out", type=Path, default=SSD_ROOT / "mx1_driver_result.json")
    args = parser.parse_args()
    if args.mem_probe_steps <= 0:
        parser.error("--mem-probe-steps must be positive")

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
    elif args.mode == "mem-probe":
        result["mem_probe"] = run_mem_probe(args)
        result["status"] = result["mem_probe"]["status"]
    result["launch_ticket"] = launch_ticket(args, smoke, mlx_probe)
    write_json(args.out, result)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    if args.mode in {"mlx-parity", "mlx-train"} and mlx_probe["status"] == "blocked":
        raise SystemExit(2)
    if args.mode == "mem-probe" and result.get("status") == "blocked":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
