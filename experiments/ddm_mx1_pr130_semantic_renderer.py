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
import inspect
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

from tac.admission_guard import assert_governed_admission
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
MX1_FIRE_GUARD_VERDICT_SCHEMA = "ddm_mx1_fire_guard_verdict.v1"
MX1B_MEM_PROBE_RESULT = REPO / ".omx/research/ddm_mx1b_20260806/mem_probe_cpu_result.json"
METAL_UNKNOWN_MARGIN_GIB = 65.0
ROW1_SAFE_RUN_RSS_MB = 90_000
ROW1_SAFE_RUN_TIMEOUT_S = 28_800
DEFAULT_WIRED_LIMIT_FRACTION = 0.35


class MemoryLimitConfigurationError(RuntimeError):
    """Raised when GPU mode cannot install a fail-closed software budget."""


class MemoryBudgetExceeded(RuntimeError):
    """Raised when measured MLX active memory plus RSS delta exceeds budget."""

    def __init__(self, message: str, *, check: dict[str, Any]) -> None:
        super().__init__(message)
        self.check = check


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _host_fingerprint() -> dict[str, str]:
    return {
        "node": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "system": platform.system(),
    }


def _gib_or_none(num_bytes: int | float | None) -> float | None:
    if num_bytes is None:
        return None
    return round(float(num_bytes) / GIB, 6)


def _safe_label_token(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_")


def _derive_row1_safe_run_projection() -> dict[str, Any]:
    measured_cpu_peak_gib: float | None = None
    source_status = "missing"
    try:
        payload = json.loads(MX1B_MEM_PROBE_RESULT.read_text(encoding="utf-8"))
        measured_cpu_peak_gib = float(
            payload["mem_probe"]["receipt"]["peak"]["peak_rss_gib"]
        )
        source_status = "read"
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        source_status = "unavailable"
    if measured_cpu_peak_gib is None:
        projected_gib = METAL_UNKNOWN_MARGIN_GIB
        arithmetic = (
            f"{METAL_UNKNOWN_MARGIN_GIB:.6f} GiB Metal/model/scorer/load-step unknown margin; "
            f"mx1b CPU-side peak unavailable at {MX1B_MEM_PROBE_RESULT}"
        )
    else:
        projected_gib = measured_cpu_peak_gib + METAL_UNKNOWN_MARGIN_GIB
        arithmetic = (
            f"{measured_cpu_peak_gib:.6f} GiB measured CPU-side peak + "
            f"{METAL_UNKNOWN_MARGIN_GIB:.6f} GiB Metal/model/scorer/load-step unknown margin = "
            f"{projected_gib:.6f} GiB"
        )
    return {
        "schema": "ddm_mx1_row1_safe_run_projection.v1",
        "axis": "[load-phase memory telemetry projection; score_claim=false]",
        "score_claim": False,
        "mx1b_mem_probe_result": str(MX1B_MEM_PROBE_RESULT),
        "mx1b_source_status": source_status,
        "measured_cpu_peak_gib": measured_cpu_peak_gib,
        "metal_unknown_margin_gib": METAL_UNKNOWN_MARGIN_GIB,
        "projected_gib": round(projected_gib, 6),
        "arithmetic": arithmetic,
        "safe_run_rss_mb": ROW1_SAFE_RUN_RSS_MB,
        "safe_run_timeout_s": ROW1_SAFE_RUN_TIMEOUT_S,
    }


def _wrap_fire_argv(raw_argv: list[str], *, label: str, projection: dict[str, Any]) -> list[str]:
    return [
        ".venv/bin/python",
        "tools/safe_run.py",
        "--rss-mb",
        str(ROW1_SAFE_RUN_RSS_MB),
        "--timeout",
        str(ROW1_SAFE_RUN_TIMEOUT_S),
        "--projected-gib",
        f"{float(projection['projected_gib']):.6f}",
        "--label",
        label,
        "--",
        *raw_argv,
    ]


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


def _system_total_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.virtual_memory().total)
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
        self._software_budget_bytes: int | None = None
        self._software_budget_required = False
        self._budget_check_count = 0
        self._budget_peak_bytes = 0
        self._last_budget_check: dict[str, Any] | None = None

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

    def install_software_budget(self, memory_limits: Mapping[str, Any]) -> None:
        budget = memory_limits.get("software_budget_bytes")
        self._software_budget_bytes = None if budget is None else int(budget)
        self._software_budget_required = bool(memory_limits.get("software_cap_required"))
        if self._software_budget_required and self._software_budget_bytes is None:
            raise MemoryLimitConfigurationError(
                "REFUSED gpu mode: software memory budget was not installed"
            )

    def sample_and_check(
        self,
        stage: str,
        *,
        mx: Any | None = None,
        note: str | None = None,
    ) -> dict[str, Any]:
        sample = self.sample(stage, mx=mx, note=note)
        self.check_budget(stage, mx=mx)
        return sample

    def check_budget(self, stage: str, *, mx: Any | None = None) -> dict[str, Any] | None:
        if self._software_budget_bytes is None:
            if self._software_budget_required:
                raise MemoryLimitConfigurationError(
                    "REFUSED gpu mode: software memory budget check requested before installation"
                )
            return None
        rss_bytes = _process_rss_bytes()
        if self._start_rss_bytes is None:
            self._start_rss_bytes = rss_bytes
        if rss_bytes is None or self._start_rss_bytes is None:
            raise MemoryLimitConfigurationError(
                "REFUSED gpu mode: software budget cannot read process RSS"
            )
        rss_delta = max(0, int(rss_bytes) - int(self._start_rss_bytes))
        mlx_active: int
        if mx is None:
            mlx_active = 0
        else:
            mlx_active_raw = _mlx_allocator_bytes(mx).get("active")
            if mlx_active_raw is None:
                raise MemoryLimitConfigurationError(
                    "REFUSED gpu mode: software budget cannot read mx.get_active_memory()"
                )
            mlx_active = int(mlx_active_raw)
        combined = int(mlx_active) + int(rss_delta)
        self._budget_check_count += 1
        self._budget_peak_bytes = max(self._budget_peak_bytes, combined)
        check = {
            "stage": stage,
            "timestamp_utc": _utc_now_iso(),
            "budget_bytes": self._software_budget_bytes,
            "budget_gib": _gib_or_none(self._software_budget_bytes),
            "mlx_active_gib": _gib_or_none(mlx_active),
            "rss_delta_from_start_gib": _gib_or_none(rss_delta),
            "combined_gib": _gib_or_none(combined),
            "within_budget": combined <= self._software_budget_bytes,
            "check_index": self._budget_check_count,
        }
        self._last_budget_check = check
        if combined > self._software_budget_bytes:
            self.sample(stage, mx=mx, note="software memory budget exceeded")
            raise MemoryBudgetExceeded(
                "software memory budget exceeded: "
                f"stage={stage} combined={combined} budget={self._software_budget_bytes}",
                check=check,
            )
        return check

    def budget_summary(self) -> dict[str, Any]:
        return {
            "enforcement": "software_stage_step_cap",
            "budget_bytes": self._software_budget_bytes,
            "budget_gib": _gib_or_none(self._software_budget_bytes),
            "required": self._software_budget_required,
            "check_count": self._budget_check_count,
            "peak_combined_gib": _gib_or_none(self._budget_peak_bytes),
            "last_check": self._last_budget_check,
            "rule": "mx.get_active_memory() + max(0, process_rss - start_process_rss) <= budget",
        }

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
        memory_probe.sample_and_check("before_selected_cache_load")
    input_tokens, input_meta = _load_selected_seg_tokens(input_cache, pair_ids)
    if memory_probe is not None:
        memory_probe.sample_and_check("after_input_cache_selected_clone")
    if input_cache == target_cache:
        target_tokens = input_tokens
        target_meta = {**input_meta, "shared_with_input_cache": True}
    else:
        target_tokens, target_meta = _load_selected_seg_tokens(target_cache, pair_ids)
        if memory_probe is not None:
            memory_probe.sample_and_check("after_target_cache_selected_clone")

    conditioning_np = input_tokens.numpy().astype(np.int32, copy=True)
    target_np = target_tokens.numpy().astype(np.int32, copy=True)
    del input_tokens
    del target_tokens
    gc.collect()
    if memory_probe is not None:
        memory_probe.sample_and_check("after_selected_cache_numpy_copy_and_torch_free")
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


def _derive_mem_budget_gb(explicit_gb: float | None, *, mem_probe: bool = False) -> dict[str, Any]:
    if explicit_gb is not None:
        if explicit_gb <= 0:
            raise ValueError("--mem-budget-gb must be positive when provided")
        return {
            "budget_gb": float(explicit_gb),
            "source": "explicit_cli",
            "mem_probe_cap_gb": 24.0 if mem_probe else None,
            "available_gib_at_start": _gib_or_none(_system_available_bytes()),
        }
    available = _system_available_bytes()
    if available is None:
        return {
            "budget_gb": None,
            "source": "unavailable_no_limit_applied",
            "mem_probe_cap_gb": 24.0 if mem_probe else None,
            "available_gib_at_start": None,
        }
    available_gib = float(available) / GIB
    default_budget = max(1.0, available_gib * 0.35)
    if mem_probe:
        default_budget = min(24.0, default_budget)
    return {
        "budget_gb": round(default_budget, 3),
        "source": "default_35pct_of_available_memory_at_start"
        if not mem_probe
        else "mem_probe_min_24gb_default_35pct_of_available_memory_at_start",
        "mem_probe_cap_gb": 24.0 if mem_probe else None,
        "available_gib_at_start": round(available_gib, 6),
    }


def _resolve_attr(obj: Any, dotted_name: str) -> Any:
    cur = obj
    for part in dotted_name.split("."):
        cur = getattr(cur, part)
    return cur


def _call_mlx_limit_with_signature(
    mx: Any,
    dotted_name: str,
    value: int,
    *,
    require_hard: bool,
    allow_soft: bool,
) -> dict[str, Any]:
    try:
        obj = _resolve_attr(mx, dotted_name)
    except AttributeError:
        return {
            "target": dotted_name,
            "status": "unavailable",
            "value_bytes": value,
            "hard_limit": False,
            "signature_form": "missing",
        }

    relaxed_supported: bool | None
    signature_text: str | None = None
    try:
        sig = inspect.signature(obj)
        signature_text = str(sig)
        relaxed_supported = (
            "relaxed" in sig.parameters
            or any(param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values())
        )
    except (TypeError, ValueError):
        relaxed_supported = None

    if relaxed_supported is not False:
        try:
            obj(value, relaxed=False)
            return {
                "target": dotted_name,
                "status": "applied",
                "value_bytes": value,
                "hard_limit": True,
                "relaxed": False,
                "signature": signature_text,
                "signature_form": "value_relaxed_false"
                if relaxed_supported is True
                else "value_relaxed_false_uninspectable",
            }
        except TypeError as exc:
            if relaxed_supported is True or (require_hard and not allow_soft):
                return {
                    "target": dotted_name,
                    "status": "failed",
                    "value_bytes": value,
                    "hard_limit": False,
                    "signature": signature_text,
                    "signature_form": "value_relaxed_false",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
        except Exception as exc:
            return {
                "target": dotted_name,
                "status": "failed",
                "value_bytes": value,
                "hard_limit": False,
                "signature": signature_text,
                "signature_form": "value_relaxed_false",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }

    if require_hard and not allow_soft:
        return {
            "target": dotted_name,
            "status": "refused_soft_only",
            "value_bytes": value,
            "hard_limit": False,
            "signature": signature_text,
            "signature_form": "value_only",
            "error": "installed MLX memory limit API has no relaxed=False hard-cap form",
        }
    try:
        obj(value)
        return {
            "target": dotted_name,
            "status": "applied_soft_allowed",
            "value_bytes": value,
            "hard_limit": False,
            "relaxed": "default",
            "signature": signature_text,
            "signature_form": "value_only",
            "soft_limit_allowed_by_cli": allow_soft,
        }
    except Exception as exc:
        return {
            "target": dotted_name,
            "status": "failed",
            "value_bytes": value,
            "hard_limit": False,
            "signature": signature_text,
            "signature_form": "value_only",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def _call_mlx_limit_value_only(mx: Any, dotted_name: str, value: int) -> dict[str, Any]:
    try:
        obj = _resolve_attr(mx, dotted_name)
    except AttributeError:
        return {
            "target": dotted_name,
            "status": "unavailable",
            "value_bytes": value,
            "hard_limit": False,
            "signature_form": "missing",
        }
    signature_text: str | None = None
    try:
        signature_text = str(inspect.signature(obj))
    except (TypeError, ValueError):
        signature_text = None
    try:
        previous = obj(value)
        return {
            "target": dotted_name,
            "status": "applied",
            "value_bytes": value,
            "previous_value_bytes": previous if isinstance(previous, int) else None,
            "hard_limit": False,
            "signature": signature_text,
            "signature_form": "value_only_soft_guideline",
        }
    except Exception as exc:
        return {
            "target": dotted_name,
            "status": "failed",
            "value_bytes": value,
            "hard_limit": False,
            "signature": signature_text,
            "signature_form": "value_only_soft_guideline",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def _derive_wired_limit_bytes(memory_limit: int) -> dict[str, Any]:
    total = _system_total_bytes()
    if total is None:
        return {
            "wired_limit": memory_limit,
            "system_total_bytes": None,
            "derived_wired_fraction": DEFAULT_WIRED_LIMIT_FRACTION,
            "source": "budget_bytes_total_memory_unavailable",
        }
    derived = int(float(total) * DEFAULT_WIRED_LIMIT_FRACTION)
    return {
        "wired_limit": min(memory_limit, derived),
        "system_total_bytes": total,
        "derived_wired_fraction": DEFAULT_WIRED_LIMIT_FRACTION,
        "source": "min_budget_bytes_35pct_total_memory",
    }


def _configure_mlx_memory_limits(
    mx: Any,
    explicit_gb: float | None,
    *,
    device: str,
    allow_soft_mem_limit: bool = False,
    mem_probe: bool = False,
    derived_budget: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    derived = dict(derived_budget or _derive_mem_budget_gb(explicit_gb, mem_probe=mem_probe))
    budget_gb = derived["budget_gb"]
    software_cap_required = str(device).lower() == "gpu"
    if budget_gb is None:
        if software_cap_required:
            raise MemoryLimitConfigurationError(
                "REFUSED gpu mode: available memory could not be read, so no software memory budget can be derived"
            )
        return {
            **derived,
            "enforcement": "software_stage_step_cap",
            "software_cap_required": software_cap_required,
            "software_cap_installed": False,
            "software_budget_bytes": None,
            "memory_limit": None,
            "cache_limit": None,
            "wired_limit": None,
            "hard_limit_required": False,
            "hard_limit_satisfied": False,
            "soft_limit_allowed_by_cli": allow_soft_mem_limit,
            "calls": [],
        }
    memory_limit = int(float(budget_gb) * GIB)
    cache_limit = int(max(256 * 1024 * 1024, memory_limit * 0.25))
    wired = _derive_wired_limit_bytes(memory_limit)
    wired_limit = int(wired["wired_limit"])
    calls = [
        _call_mlx_limit_value_only(mx, "set_memory_limit", memory_limit),
        _call_mlx_limit_value_only(mx, "set_cache_limit", cache_limit),
        _call_mlx_limit_value_only(mx, "set_wired_limit", wired_limit),
    ]
    software_cap_installed = memory_limit > 0
    if software_cap_required and not software_cap_installed:
        raise MemoryLimitConfigurationError(
            "REFUSED gpu mode: software memory budget was not installed"
        )
    return {
        **derived,
        "enforcement": "software_stage_step_cap",
        "software_cap_required": software_cap_required,
        "software_cap_installed": software_cap_installed,
        "software_budget_bytes": memory_limit,
        "software_budget_rule": "mx.get_active_memory() + max(0, process_rss - start_process_rss) <= budget",
        "memory_limit": memory_limit,
        "cache_limit": cache_limit,
        "wired_limit": wired_limit,
        "wired_limit_derivation": wired,
        "wired_limit_semantics": (
            "MLX 0.31.2 set_wired_limit limits memory kept resident on macOS 15+; "
            "set_memory_limit is only a graph-evaluation guideline and is not a hard allocation cap."
        ),
        "hard_limit_required": False,
        "hard_limit_satisfied": False,
        "hard_limit_deprecated_reason": "MLX 0.31.2 set_memory_limit(limit) is soft and has no relaxed=False API.",
        "soft_limit_allowed_by_cli": allow_soft_mem_limit,
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
    memory_probe.sample_and_check(stage, mx=mx, note=note)


def run_mlx_train(
    args: argparse.Namespace,
    *,
    memory_probe: LoadPhaseMemoryProbe | None = None,
) -> dict[str, Any]:
    """Run the real MLX Row-1 training path when MLX is available."""

    probe = memory_probe if memory_probe is not None else LoadPhaseMemoryProbe()
    mem_probe_mode = getattr(args, "mode", "") == "mem-probe"
    budget_plan = _derive_mem_budget_gb(args.mem_budget_gb, mem_probe=mem_probe_mode)
    probe.install_software_budget({
        **budget_plan,
        "software_cap_required": str(args.device).lower() == "gpu",
        "software_budget_bytes": None
        if budget_plan["budget_gb"] is None
        else int(float(budget_plan["budget_gb"]) * GIB),
    })
    probe.sample_and_check("start")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    pair_ids = _select_stratified_indices(args.pairs, seed=args.seed)
    checkpoint = torch.load(args.init, map_location="cpu", weights_only=False)
    probe.sample_and_check("after_init_checkpoint_torch_load")
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
    try:
        memory_limits = _configure_mlx_memory_limits(
            mx,
            args.mem_budget_gb,
            device=args.device,
            allow_soft_mem_limit=bool(getattr(args, "allow_soft_mem_limit", False)),
            mem_probe=mem_probe_mode,
            derived_budget=budget_plan,
        )
        probe.install_software_budget(memory_limits)
    except Exception as exc:
        probe.sample_and_check(
            "after_require_mlx_memory_limit_configuration_failed",
            mx=mx,
            note=f"{type(exc).__name__}: {exc}",
        )
        raise
    probe.sample_and_check("after_require_mlx_and_memory_limits", mx=mx)
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
        probe.sample_and_check("after_resume_load_and_init_checkpoint_free", mx=mx)
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
        probe.sample_and_check("after_torch_checkpoint_free", mx=mx)

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
    probe.sample_and_check("after_selected_token_numpy_free", mx=mx)

    segnet_torch = _load_upstream_segnet(torch.device("cpu"))
    probe.sample_and_check("after_upstream_segnet_torch_load", mx=mx)
    segnet_mlx = torch_segnet_to_mlx(segnet_torch)
    segnet_params = segnet_mlx.parameters() if hasattr(segnet_mlx, "parameters") else []
    _mx_eval_setup_barrier(mx, probe, "after_segnet_mlx_conversion", segnet_params)
    del segnet_torch
    gc.collect()
    _clear_mlx_cache(mx)
    probe.sample_and_check("after_segnet_torch_free", mx=mx)
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
        probe.check_budget(f"after_train_step_{step + 1:06d}", mx=mx)
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
            probe.check_budget(f"after_eval_step_{step + 1:06d}", mx=mx)
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
                probe.sample_and_check(f"after_checkpoint_step_{step + 1:06d}", mx=mx)
    elapsed = time.time() - start_time
    latest_path = args.run_dir / "mlx.latest.npz"
    resume_check = load_stage_checkpoint_npz(latest_path, model=model, optimizer=optimizer, mx=mx)
    probe.sample_and_check("after_resume_check", mx=mx)
    return {
        "schema": "ddm_mx1_mlx_train.v1",
        "status": "passed",
        "axis": "[macOS-MLX research-signal]",
        "score_claim": False,
        "pairs": pair_ids,
        "cache_load": cache_meta,
        "memory_limits": memory_limits,
        "software_budget": probe.budget_summary(),
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
    payload["mode"] = "mem-probe"
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
    software_budget = (
        train_result.get("software_budget")
        if train_result is not None and isinstance(train_result.get("software_budget"), dict)
        else probe.budget_summary()
    )
    software_last = software_budget.get("last_check")
    software_budget_clearance = (
        int(software_budget.get("check_count") or 0) >= int(probe_args.steps)
        and isinstance(software_last, dict)
        and software_last.get("within_budget") is True
    )
    metal_fire_clearance = (
        status == "passed"
        and final_step_sample is not None
        and has_mlx_allocator_telemetry
        and software_budget_clearance
    )
    return {
        "schema": MEM_PROBE_RECEIPT_SCHEMA,
        "status": status,
        "axis": "[load-phase memory telemetry; score_claim=false]",
        "score_claim": False,
        "source_repo_head": SOURCE_REPO_HEAD,
        "source_repo_root": SOURCE_REPO_ROOT,
        "host": _host_fingerprint(),
        "mode": "mem-probe",
        "device_request": args.device,
        "pairs": int(args.pairs),
        "pair_ids": _select_stratified_indices(args.pairs, seed=args.seed),
        "requested_training_steps": int(probe_args.steps),
        "mem_budget_gb_arg": args.mem_budget_gb,
        "input_cache": str(args.input_cache),
        "target_cache": str(args.target_cache),
        "init_checkpoint": str(args.init),
        "argv_config": {
            "device": args.device,
            "pairs": int(args.pairs),
            "lr": float(args.lr),
            "ce_fraction": float(args.ce_fraction),
            "softplus_fraction": float(args.softplus_fraction),
            "bits": int(args.bits),
            "mem_budget_gb": args.mem_budget_gb,
            "allow_soft_mem_limit": bool(getattr(args, "allow_soft_mem_limit", False)),
            "input_cache": str(args.input_cache),
            "target_cache": str(args.target_cache),
            "init": str(args.init),
        },
        "memory_limits": None if train_result is None else train_result.get("memory_limits"),
        "software_budget": software_budget,
        "samples": probe.samples,
        "peak": probe.peak(),
        "train_result_summary": None
        if train_result is None
        else {
            "status": train_result.get("status"),
            "steps": train_result.get("steps"),
            "seconds_per_step": train_result.get("seconds_per_step"),
            "memory_limits": train_result.get("memory_limits"),
            "stage_checkpoint": train_result.get("stage_checkpoint"),
            "latest_checkpoint": train_result.get("latest_checkpoint"),
            "latest_checkpoint_sha256": train_result.get("latest_checkpoint_sha256"),
            "load_memory_peak": train_result.get("load_memory_peak"),
            "software_budget": train_result.get("software_budget"),
        },
        "blocker": blocker,
        "clearance_checks": {
            "required_stage": required_stage,
            "has_required_stage_sample": final_step_sample is not None,
            "has_mlx_allocator_telemetry_at_required_stage": has_mlx_allocator_telemetry,
            "software_budget_check_count": software_budget.get("check_count"),
            "software_budget_within_limit": None
            if not isinstance(software_last, dict)
            else software_last.get("within_budget"),
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
    budget_exc: MemoryBudgetExceeded | None = None
    try:
        train_result = run_mlx_train(probe_args, memory_probe=probe)
        status = str(train_result.get("status", "passed"))
    except Exception as exc:
        status = "blocked" if isinstance(exc, MlxUnavailableError) else "failed"
        if isinstance(exc, MemoryBudgetExceeded):
            budget_exc = exc
        last_sample = probe.samples[-1] if probe.samples else None
        blocker = {
            "error_type": type(exc).__name__,
            "error": str(exc),
            "last_sample_stage": None if last_sample is None else last_sample.get("stage"),
            "sample_count": len(probe.samples),
            "software_budget": probe.budget_summary(),
            "boundary": (
                "CPU load-path telemetry may be present before this blocker; "
                "full MLX allocator/three-step telemetry requires MAIN Metal."
            ),
        }
        if isinstance(exc, MlxUnavailableError):
            blocker["verdict_scope"] = "ENVIRONMENT: local sandbox MLX/Metal unavailable"
        elif isinstance(exc, MemoryLimitConfigurationError):
            blocker["verdict_scope"] = "INSTANCE: MLX software memory-budget configuration"
        elif isinstance(exc, MemoryBudgetExceeded):
            blocker["verdict_scope"] = "INSTANCE: software stage/step memory budget"
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
    write_json_atomic(receipt_path, receipt)
    result = {
        "schema": "ddm_mx1_mem_probe.v1",
        "status": status,
        "score_claim": False,
        "receipt_path": str(receipt_path),
        "receipt_sha256": _sha256_file(receipt_path),
        "receipt": receipt,
    }
    if budget_exc is not None:
        raise budget_exc
    return result


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
    cap_cache = args.target_cache  # GT labels as tokens AND targets
    veh_cache = args.input_cache   # public-wire (tq1c) labels as tokens, GT targets
    ticket_path = _ticket_path_for_args(args)

    # RR3-F1: a Row-1 verdict requires TWO arms — ARM-CAP (GT tokens -> GT targets, receiver
    # CAPACITY vs the fp1 flat-paint floor and PR130's external number) and ARM-VEH (public-wire
    # tq1c tokens -> GT targets, composed-vehicle correction reach). A single-arm ticket
    # conflates the two questions, so the bare argv_n32/argv_n120 keys no longer exist.
    def _arm_argv(pairs: int, seed: int, input_cache: Path, subdir: str, argv_key: str) -> list[str]:
        run_dir = args.run_dir / subdir
        fire_guard_verdict = run_dir / "fire_guard_verdict.json"
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
            "--fire-guard-verdict", str(fire_guard_verdict),
            "--launch-ticket-path", str(ticket_path),
            "--fire-argv-key", argv_key,
        ]
        if args.mem_budget_gb is not None:
            argv.extend(["--mem-budget-gb", str(args.mem_budget_gb)])
        if getattr(args, "allow_soft_mem_limit", False):
            argv.append("--allow-soft-mem-limit")
        return argv

    safe_run_projection = _derive_row1_safe_run_projection()

    def _fire_argv(pairs: int, seed: int, input_cache: Path, subdir: str, argv_key: str) -> list[str]:
        raw = _arm_argv(pairs, seed, input_cache, subdir, argv_key)
        return _wrap_fire_argv(
            raw,
            label=f"ddm_mx1_row1_{_safe_label_token(subdir)}",
            projection=safe_run_projection,
        )

    arm_specs = {
        "argv_n32_arm_cap": (32, args.seed, cap_cache, "launch_arm_cap/n32_metal"),
        "argv_n32_arm_veh": (32, args.seed, veh_cache, "launch_arm_veh/n32_metal"),
        "argv_n120_arm_cap": (120, args.seed + 1, cap_cache, "launch_arm_cap/n120_metal"),
        "argv_n120_arm_veh": (120, args.seed + 1, veh_cache, "launch_arm_veh/n120_metal"),
    }
    mem_probe_receipt_paths = {
        key: str(args.run_dir / subdir / "mem_probe" / "mem_probe_receipt.json")
        for key, (_pairs, _seed, _cache, subdir) in arm_specs.items()
    }

    def _mem_probe_command(pairs: int, seed: int, input_cache: Path, subdir: str) -> list[str]:
        probe_run_dir = args.run_dir / subdir / "mem_probe"
        command = [
            ".venv/bin/python",
            "experiments/ddm_mx1_pr130_semantic_renderer.py",
            "--mode", "mem-probe",
            "--device", "gpu",
            "--pairs", str(pairs),
            "--mem-probe-steps", str(args.mem_probe_steps),
            "--lr", str(args.lr),
            "--ce-fraction", str(args.ce_fraction),
            "--softplus-fraction", str(args.softplus_fraction),
            "--bits", str(args.bits),
            "--seed", str(seed),
            "--checkpoint-every", str(max(1, int(args.mem_probe_steps))),
            "--eval-every", "1",
            "--input-cache", str(input_cache),
            "--target-cache", str(args.target_cache),
            "--init", str(args.init),
            "--run-dir", str(probe_run_dir),
            "--out", str(probe_run_dir / "mem_probe_result.json"),
            "--launch-ticket-path", str(ticket_path),
        ]
        if args.mem_budget_gb is not None:
            command.extend(["--mem-budget-gb", str(args.mem_budget_gb)])
        if getattr(args, "allow_soft_mem_limit", False):
            command.append("--allow-soft-mem-limit")
        return command

    mem_probe_commands = {
        key: _mem_probe_command(pairs, seed, cache, subdir)
        for key, (pairs, seed, cache, subdir) in arm_specs.items()
    }
    mem_probe_receipt_path = Path(mem_probe_receipt_paths["argv_n32_arm_cap"])
    mem_probe_command = mem_probe_commands["argv_n32_arm_cap"]
    fire_guard_verdict_paths = {
        key: str(args.run_dir / subdir / "fire_guard_verdict.json")
        for key, (_pairs, _seed, _cache, subdir) in arm_specs.items()
    }
    fire_guard_commands = {
        key: [
            ".venv/bin/python",
            "tools/mx1_fire_guard.py",
            "--ticket", str(ticket_path),
            "--argv-key", key,
            "--out", fire_guard_verdict_paths[key],
        ]
        for key in arm_specs
    }
    fire_argvs = {
        key: _fire_argv(pairs, seed, cache, subdir, key)
        for key, (pairs, seed, cache, subdir) in arm_specs.items()
    }

    return {
        "schema": "ddm_mx1_row1_launch_ticket.v4_software_cap_fire_guarded",
        "score_claim": False,
        "launch_ticket_path": str(ticket_path),
        "mem_probe_receipt_required": True,
        "mem_probe_receipt_path": str(mem_probe_receipt_path),
        "mem_probe_receipt_paths": mem_probe_receipt_paths,
        "mem_probe_command": mem_probe_command,
        "mem_probe_commands": mem_probe_commands,
        "fire_guard_required": True,
        "fire_guard_tool": "tools/mx1_fire_guard.py",
        "fire_guard_verdict_schema": MX1_FIRE_GUARD_VERDICT_SCHEMA,
        "fire_guard_verdict_paths": fire_guard_verdict_paths,
        "fire_guard_commands": fire_guard_commands,
        "main_fire_sequence": [
            {
                "step": "guard_precheck",
                "command": fire_guard_commands["argv_n32_arm_cap"],
                "expected": "REFUSE until the matching mem_probe_receipt exists and passes",
            },
            {
                "step": "probe",
                "command": mem_probe_command,
                "expected": "writes mem_probe_receipt.json atomically with status=passed",
            },
            {
                "step": "gate",
                "command": fire_guard_commands["argv_n32_arm_cap"],
                "expected": "writes fire_guard_verdict.json with status=passed",
            },
            {
                "step": "fire",
                "command": fire_argvs["argv_n32_arm_cap"],
                "expected": "entrypoint re-runs tools.mx1_fire_guard against --launch-ticket-path/--fire-argv-key before MLX setup",
            },
        ],
        "scheduling": (
            "SEQUENTIAL one-Metal-fire-at-a-time — operator machine OOM 2026-08-06; "
            "ARM-VEH fires only after ARM-CAP completes or a composed measured-peak "
            "projection shows headroom under 116GiB"
        ),
        "fire_protocol": {
            "pre_fire_liveness_proof": (
                "A SUCCESSFUL enumerator is required before every fire. If pgrep returns rc>=2, "
                "run ps axo command; if ps also fails or is denied (rc!=0), REFUSE. Never map "
                "a denied enumerator to 0 candidates."
            ),
            "rr8_f1_refuse_condition": "pgrep rc>=2 AND ps rc!=0",
            "anti_pattern": "no `|| true` around the fallback enumerator; denied ps is not quiescence",
            "scheduling": "SEQUENTIAL one-Metal-fire-at-a-time",
        },
        "safe_run_projection": safe_run_projection,
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
        "argv_n32_arm_cap": fire_argvs["argv_n32_arm_cap"],
        "argv_n32_arm_veh": fire_argvs["argv_n32_arm_veh"],
        "argv_n120_arm_cap": fire_argvs["argv_n120_arm_cap"],
        "argv_n120_arm_veh": fire_argvs["argv_n120_arm_veh"],
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
            "mem_budget_default_policy": "software cap at 35% of available memory at process start when --mem-budget-gb is omitted; mem-probe caps default at min(24GB, default)",
            "enforcement": "software_stage_step_cap",
            "software_budget_rule": "mx.get_active_memory() + max(0, process_rss - start_process_rss) <= budget",
            "wired_limit_policy": "attempt mx.set_wired_limit(min(budget, 35% of system total)) when available",
            "allow_soft_mem_limit": bool(getattr(args, "allow_soft_mem_limit", False)),
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


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(tmp, path)


def _ticket_path_for_args(args: argparse.Namespace) -> Path:
    explicit = getattr(args, "launch_ticket_path", None)
    if explicit is not None:
        return Path(explicit)
    return args.run_dir / "launch_ticket_v4_fire_guarded.json"


def _canonical_existing_or_repo_path(path: Path | str | None) -> str | None:
    if path is None:
        return None
    resolved = Path(path).expanduser()
    resolved = (REPO / resolved).resolve() if not resolved.is_absolute() else resolved.resolve()
    return str(resolved)


def _assert_gpu_fire_guard(args: argparse.Namespace) -> None:
    verdict_path = args.fire_guard_verdict
    ticket_path = args.launch_ticket_path
    argv_key = args.fire_argv_key
    if verdict_path is None or ticket_path is None or not argv_key:
        print(
            "[mx1-fire-guard] REFUSED: gpu mlx-train requires --fire-guard-verdict, "
            "--launch-ticket-path, and --fire-argv-key",
            file=sys.stderr,
        )
        raise SystemExit(9)
    try:
        from tools.mx1_fire_guard import evaluate_guard

        evaluated = evaluate_guard(ticket_path, argv_key)
    except Exception as exc:
        print(
            f"[mx1-fire-guard] REFUSED: in-process guard evaluation failed for "
            f"{ticket_path} {argv_key}: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(9) from exc
    if evaluated.get("schema") != MX1_FIRE_GUARD_VERDICT_SCHEMA or evaluated.get("status") != "passed":
        print(
            f"[mx1-fire-guard] REFUSED: in-process guard failed: "
            f"status={evaluated.get('status')!r} reason={evaluated.get('reason_code')!r}",
            file=sys.stderr,
        )
        raise SystemExit(9)
    expected_verdict_path = (evaluated.get("fire_config") or {}).get("fire_guard_verdict")
    if _canonical_existing_or_repo_path(expected_verdict_path) != _canonical_existing_or_repo_path(verdict_path):
        print(
            "[mx1-fire-guard] REFUSED: --fire-guard-verdict does not match ticket fire argv "
            f"expected={expected_verdict_path!r} actual={str(verdict_path)!r}",
            file=sys.stderr,
        )
        raise SystemExit(9)
    try:
        verdict = json.loads(verdict_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(
            f"[mx1-fire-guard] REFUSED: could not read guard verdict at {verdict_path}: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(9) from exc
    if (
        verdict.get("schema") != MX1_FIRE_GUARD_VERDICT_SCHEMA
        or verdict.get("status") != "passed"
        or verdict.get("reason_code") != "fire_guard_passed"
    ):
        print(
            f"[mx1-fire-guard] REFUSED: guard verdict failed or malformed at {verdict_path}: "
            f"status={verdict.get('status')!r} reason={verdict.get('reason_code')!r}",
            file=sys.stderr,
        )
        raise SystemExit(9)
    for key in ("ticket_path", "receipt_path"):
        if _canonical_existing_or_repo_path(verdict.get(key)) != _canonical_existing_or_repo_path(evaluated.get(key)):
            print(
                f"[mx1-fire-guard] REFUSED: guard verdict {key} does not match fresh evaluation",
                file=sys.stderr,
            )
            raise SystemExit(9)
    if verdict.get("argv_key") != argv_key:
        print(
            "[mx1-fire-guard] REFUSED: guard verdict argv_key does not match current fire argv",
            file=sys.stderr,
        )
        raise SystemExit(9)


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
    parser.add_argument("--allow-soft-mem-limit", action="store_true")
    parser.add_argument("--fire-guard-verdict", type=Path)
    parser.add_argument("--launch-ticket-path", type=Path)
    parser.add_argument("--fire-argv-key")
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--out", type=Path, default=SSD_ROOT / "mx1_driver_result.json")
    args = parser.parse_args()
    if args.mem_probe_steps <= 0:
        parser.error("--mem-probe-steps must be positive")
    if args.mode in {"mlx-train", "torch-smoke"}:
        assert_governed_admission(f"ddm_mx1_pr130_semantic_renderer:{args.mode}")
    if args.mode == "mlx-train" and str(args.device).lower() == "gpu":
        _assert_gpu_fire_guard(args)

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
    write_json(_ticket_path_for_args(args), result["launch_ticket"])
    write_json(args.out, result)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    if args.mode in {"mlx-parity", "mlx-train"} and mlx_probe["status"] == "blocked":
        raise SystemExit(2)
    if args.mode == "mem-probe" and result.get("status") != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
