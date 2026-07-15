#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Torch/CUDA training entry point for the typed V9 CGauge #432 program.

The production path always derives configuration from ``spec_v9_cgauge``; arbitrary
hand-written trainer flags are intentionally not accepted.  This is a real frozen-scorer
gradient path (SegNet and PoseNet consume the rendered frames through R), not a cached-label
surrogate.  Local ``--verify-only`` is CPU-light and never instantiates the scorers.

Authority remains unchanged: training rows are ``[contest-CUDA training-advisory]`` and
NON-PROMOTABLE.  Only byte-closed ``upstream/evaluate.py`` results can move the pointer.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import sys
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "src", REPO / "experiments", REPO / "upstream"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from tac.boundary_math.lever_b_levelset_generator import (
    CurveletBankConfig,
    build_coords,
    build_static_core_phi_target,
    curvelet_directional_B,
    curvelet_feats,
)
from tac.cuda_levelset_training import (
    NUMPY_FP32_PARITY_COSINE_BAR,
    CudaLevelSetConfig,
    DeterministicPairCursor,
    TorchLevelSetWitness,
    TorchPoseCarrier,
    apply_torch_execution_policy,
    area_constraint_torch,
    chroma_boundary_loss,
    clip_grad_groups,
    compile_identity_probe,
    contest_r,
    eikonal_and_length,
    forward_parity_against_numpy,
    homography_grid_from_xi,
    island_birth_from_signed_torch,
    island_birth_perclass_from_signed_torch,
    parameter_groups,
    persistence_topology_loss_torch,
    pose_objective_torch,
    realized_signed_margin,
    round_ste,
    select_torch_execution_policy,
    structured_sdf_prefit,
    warp_field_persist_torch,
    weight_entropy_rate_term_torch,
    witness_tie_coordinate_torch,
)
from tac.cuda_v9_controller_runtime import (
    TorchV9ControllerRuntime,
    torch_pose_jacobian_conditioning,
)
from tac.cuda_v9_island_runtime import (
    TorchIslandTargetRuntime,
    birth_scaled_logit_offsets,
    seed_compose_weight_at_epoch,
)
from tac.cuda_v9_optimizers import TorchMuonAdamW, build_torch_muon_adamw
from tac.cuda_v9_throughput import adopt_compiled_training_region
from tac.witness_control.tail_cycles import TailController, TailCycleConfig
from tac.witness_run_artifacts import (
    TORCH_EMA_PT,
    TORCH_RESUME_PT,
    TORCH_RUN_MANIFEST_JSON,
    TORCH_TRAIN_RESULT_JSON,
    TORCH_TRAJECTORY_JSONL,
)
from tac.witness_training_contract import (
    cuda_v9_port_receipt,
    curriculum_stage,
    loss_terms_row,
)

_FORBIDDEN_TMP = ("/tmp/", "/var/tmp/", "/private/tmp/", "/private/var/tmp/")
_TORCH_POLYAK_PT = "levelset_witness_polyak_torch.pt"


def _utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _flag_map(argv: tuple[str, ...]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    i = 2  # python + trainer path
    while i < len(argv):
        tok = argv[i]
        if not tok.startswith("--"):
            i += 1
            continue
        if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
            out[tok] = argv[i + 1]
            i += 2
        else:
            out[tok] = True
            i += 1
    return out


# Named DSL config selection (runtime plumbing only: each name resolves to a
# typed spec_v9_cgauge factory whose typed hash is the scientific identity).
# "v9_cgauge_432_smoke_regime" is the #438 CUDA timing-smoke cost-regime
# variant: NON-PROMOTABLE timing evidence, never a science arm.
V9_DSL_CONFIG_CHOICES = ("v9_cgauge_432_launch", "v9_cgauge_432_smoke_regime")


def derive_config(args) -> tuple[dict[str, Any], str, tuple[str, ...]]:
    from tac.witness_dsl.spec_v9_cgauge import (
        compile_v9_cgauge_432_launch_config,
        compile_v9_cgauge_432_smoke_regime_config,
    )

    factory = {
        "v9_cgauge_432_launch": compile_v9_cgauge_432_launch_config,
        "v9_cgauge_432_smoke_regime": compile_v9_cgauge_432_smoke_regime_config,
    }[getattr(args, "dsl_config", "v9_cgauge_432_launch")]
    compiled = factory(
        args.gt_cache, num_pairs=args.num_pairs, epochs=args.epochs, out_dir=args.out_dir
    )
    argv = tuple(compiled.typed.to_program().compile_trainer_argv())
    flags = _flag_map(argv)
    payload = json.dumps({"argv": argv, "typed_config_hash": compiled.typed.typed_config_hash()}, sort_keys=True)
    return flags, hashlib.sha256(payload.encode()).hexdigest(), argv


def build_parser() -> argparse.ArgumentParser:
    """Build the runtime parser; runtime controls never enter the typed DSL hash."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-cache", default="experiments/results/mlx_fleet_gt_cache/gt_n600.npz")
    ap.add_argument("--num-pairs", type=int, default=600)
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--out-dir", default="experiments/results/v9_cgauge_cuda")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--resume-from")
    ap.add_argument("--stop-after-epochs", type=int)
    ap.add_argument("--no-implicit-resume", action="store_true")
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--expected-segnet-sha256")
    ap.add_argument("--expected-posenet-sha256")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--compile-probe", action="store_true")
    ap.add_argument(
        "--torch-compile-mode",
        choices=("off", "default", "max-autotune"),
        default=None,
        help=(
            "BACKEND runtime override for the Inductor compile mode (never "
            "typed-DSL science). MEASURED 2026-07-15 H100 r5 rc=124: "
            "max-autotune kernel benchmarking consumed the entire 1440s "
            "time-boxed window before one epoch — time-boxed smokes use "
            "'default'; long runs amortize max-autotune. Default None keeps "
            "the policy's own selection."
        ),
    )
    ap.add_argument(
        "--dsl-config",
        choices=V9_DSL_CONFIG_CHOICES,
        default="v9_cgauge_432_launch",
        help=(
            "Named typed-DSL config to compile (default: the science launch "
            "config; smoke_regime is #438 NON-PROMOTABLE timing-cost evidence)"
        ),
    )
    return ap


def _runtime_epoch_window(
    completed_epoch: int, typed_total_epochs: int, stop_after_epochs: int | None
) -> tuple[int, int]:
    """Return the inclusive next/end epochs without changing the typed horizon."""
    completed_epoch = int(completed_epoch)
    typed_total_epochs = int(typed_total_epochs)
    if typed_total_epochs <= 0:
        raise ValueError("epochs must be positive")
    if completed_epoch < 0 or completed_epoch > typed_total_epochs:
        raise ValueError("checkpoint epoch is outside the typed training horizon")
    if completed_epoch == typed_total_epochs:
        raise ValueError("checkpoint already completed the typed horizon; refusing zero-work run")
    if stop_after_epochs is not None and not 1 <= int(stop_after_epochs) <= 3:
        raise ValueError("stop-after-epochs must be in 1..3")
    additional = (
        typed_total_epochs - completed_epoch
        if stop_after_epochs is None
        else int(stop_after_epochs)
    )
    return completed_epoch + 1, min(typed_total_epochs, completed_epoch + additional)


def _canonical_checkpoint_due(
    epoch: int,
    *,
    ckpt_every: int,
    run_end_epoch: int,
    runtime_stop_after_epochs: int | None,
    tail_stop_after_epoch: bool,
) -> bool:
    """Return whether the completed epoch must refresh the full resume state."""
    return bool(
        runtime_stop_after_epochs is not None
        or int(epoch) % int(ckpt_every) == 0
        or int(epoch) == int(run_end_epoch)
        or tail_stop_after_epoch
    )


def _resolve_resume_intent(
    out: Path, resume_from: str | None, *, no_implicit_resume: bool
) -> Path | None:
    """Resolve strict remote intent or the legacy canonical auto-resume path."""
    if resume_from:
        path = Path(resume_from)
        if path.is_dir():
            path = path / TORCH_RESUME_PT
        if not path.is_file():
            raise ValueError(f"explicit resume checkpoint does not exist: {path}")
        return path
    if not no_implicit_resume:
        legacy_path = out / TORCH_RESUME_PT
        return legacy_path if legacy_path.is_file() else None
    allowed_preflight_files = {"remote_asset_custody.json"}
    unexpected = (
        sorted(
            path.name
            for path in out.iterdir()
            if path.name not in allowed_preflight_files
            or not path.is_file()
            or path.is_symlink()
        )
        if out.exists()
        else []
    )
    if unexpected:
        raise ValueError(
            "out-dir is not fresh and --resume-from was not supplied; "
            "implicit resume/overwrite is forbidden; unexpected entries: "
            + ", ".join(unexpected)
        )
    return None


def _load_validated_gt_cache(path: str, num_pairs: int) -> dict[str, np.ndarray]:
    """Load and validate all scorer-authority cache surfaces before CUDA setup."""
    cache_path = Path(path)
    if not cache_path.is_file():
        raise ValueError(f"GT cache does not exist: {cache_path}")
    required = ("n_pairs", "lstars", "margins", "gt_f1", "gt_poses")
    with np.load(cache_path, allow_pickle=False) as z:
        missing = [key for key in required if key not in z.files]
        if missing:
            raise ValueError("GT cache is missing required keys: " + ", ".join(missing))
        declared = int(np.asarray(z["n_pairs"]).reshape(()))
        arrays = {key: np.asarray(z[key]) for key in required if key != "n_pairs"}
    if declared < num_pairs:
        raise ValueError("GT cache contains fewer pairs than requested")
    lstars = arrays["lstars"]
    margins = arrays["margins"]
    gt_f1 = arrays["gt_f1"]
    gt_poses = arrays["gt_poses"]
    if lstars.ndim != 3 or margins.shape != lstars.shape:
        raise ValueError("GT lstars/margins must have identical (pairs,H,W) geometry")
    if gt_f1.ndim != 4 or gt_f1.shape[-1] != 3:
        raise ValueError("GT gt_f1 must have (pairs,H,W,3) geometry")
    if gt_poses.ndim != 2 or gt_poses.shape[1] != 6:
        raise ValueError("GT gt_poses must have (pairs,6) geometry")
    for key, value in arrays.items():
        if value.shape[0] < num_pairs:
            raise ValueError(f"GT cache {key} contains fewer pairs than requested")
    return {key: value[:num_pairs] for key, value in arrays.items()}


def _validate_gt_geometry(
    gt: Mapping[str, np.ndarray], flags: Mapping[str, Any]
) -> dict[str, list[int]]:
    """Validate GT arrays against the exact typed scorer and receiver geometry."""
    render_hw = (int(flags["--render-h"]), int(flags["--render-w"]))
    camera_hw = (CudaLevelSetConfig.camera_h, CudaLevelSetConfig.camera_w)
    if tuple(gt["lstars"].shape[1:]) != render_hw:
        raise ValueError(
            f"GT scorer geometry {tuple(gt['lstars'].shape[1:])} differs from "
            f"typed render geometry {render_hw}"
        )
    if tuple(gt["gt_f1"].shape[1:3]) != camera_hw:
        raise ValueError(
            f"GT camera geometry {tuple(gt['gt_f1'].shape[1:3])} differs from "
            f"Torch receiver {camera_hw}"
        )
    return {"render_hw": list(render_hw), "camera_hw": list(camera_hw)}


def _validate_scorer_custody(
    paths: Mapping[str, Path] | None = None,
    expected_sha256: Mapping[str, str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Validate frozen scorer bytes without constructing either scorer network."""
    from safetensors import safe_open

    scorer_paths = dict(
        paths
        or {
            "segnet": REPO / "upstream" / "models" / "segnet.safetensors",
            "posenet": REPO / "upstream" / "models" / "posenet.safetensors",
        }
    )
    if set(scorer_paths) != {"segnet", "posenet"}:
        raise ValueError("scorer custody requires exactly segnet and posenet paths")
    expected = dict(expected_sha256 or {})
    if expected and set(expected) != {"segnet", "posenet"}:
        raise ValueError("expected scorer SHA custody requires both segnet and posenet")
    receipt: dict[str, dict[str, Any]] = {}
    for name, raw_path in scorer_paths.items():
        path = Path(raw_path)
        if path.is_symlink() or not path.is_file():
            raise ValueError(
                f"canonical {name} safetensors must be a regular non-symlink file: {path}"
            )
        digest = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        actual_sha256 = digest.hexdigest()
        expected_digest = expected.get(name)
        if expected_digest is not None and actual_sha256 != expected_digest.lower():
            raise ValueError(
                f"canonical {name} safetensors SHA-256 mismatch: "
                f"{actual_sha256} != {expected_digest.lower()}"
            )
        try:
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                keys = list(handle.keys())
        except Exception as exc:
            raise ValueError(f"canonical {name} safetensors is not parseable: {path}") from exc
        if not keys:
            raise ValueError(f"canonical {name} safetensors contains no tensors: {path}")
        receipt[name] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": actual_sha256,
            "tensor_count": len(keys),
            "expected_sha256": expected_digest,
            "sha_authority": (
                "PLAN_EXPECTED_MATCH"
                if expected_digest is not None
                else "MEASURED_ONLY_legacy_non_strict"
            ),
        }
    return receipt


def _load_validated_resume(
    path: Path,
    expected_hash: str,
    total_epochs: int,
    *,
    expected_scorer_sha256: Mapping[str, str] | None = None,
    require_scorer_custody: bool = False,
) -> dict[str, Any]:
    """Validate resume custody/config on CPU before any CUDA model construction."""
    import torch

    blob = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(blob, dict):
        raise ValueError("resume checkpoint must contain a mapping")
    required = {
        "schema",
        "epoch",
        "model",
        "ema",
        "optimizer",
        "torch_rng",
        "numpy_rng",
        "python_rng",
        "config_hash",
        "dsl_argv",
    }
    missing = sorted(required - blob.keys())
    if missing:
        raise ValueError("resume checkpoint is missing required state: " + ", ".join(missing))
    if blob["schema"] != "v9_cgauge_torch_resume_v2":
        raise ValueError(f"unsupported resume checkpoint schema: {blob['schema']!r}")
    if blob.get("config_hash") != expected_hash:
        raise ValueError("resume config hash differs from the typed V9 CGauge program; refusing drift")
    expected_scorers = {
        name: value.lower() for name, value in dict(expected_scorer_sha256 or {}).items()
    }
    checkpoint_scorers = blob.get("scorer_sha256")
    if checkpoint_scorers is None:
        if require_scorer_custody:
            raise ValueError("strict resume checkpoint is missing scorer SHA-256 custody")
    else:
        if not isinstance(checkpoint_scorers, Mapping) or set(checkpoint_scorers) != {
            "segnet",
            "posenet",
        }:
            raise ValueError("resume scorer SHA-256 custody must contain segnet and posenet")
        normalized_checkpoint_scorers = {
            name: str(value).lower() for name, value in checkpoint_scorers.items()
        }
        if expected_scorers and normalized_checkpoint_scorers != expected_scorers:
            raise ValueError(
                "resume scorer SHA-256 custody differs from the strict execution plan"
            )
    epoch = int(blob.get("epoch", -1))
    if epoch < 0 or epoch > int(total_epochs):
        raise ValueError("resume checkpoint epoch is outside the typed training horizon")
    return blob


def _atomic_torch_save(obj, path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _atomic_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True))
    os.replace(tmp, path)


def _emit_trajectory_row(out: Path, row: Mapping[str, Any]) -> None:
    """Durably append one JSON telemetry row and mirror it to stdout."""
    payload = dict(row)
    with open(out / TORCH_TRAJECTORY_JSONL, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, sort_keys=True) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    print(json.dumps(payload), flush=True)


def _flush_trajectory_rows(out: Path, rows: list[dict[str, Any]]) -> None:
    """Durably emit rows whose matching canonical checkpoint already landed."""
    for row in rows:
        _emit_trajectory_row(out, row)
    rows.clear()


def _stage_epoch_row(rows: list[dict[str, Any]], row: dict[str, Any]) -> None:
    """Buffer a row for the checkpoint-consistent flush AND mirror it to stdout NOW.

    r6 pre-fire review (2026-07-15, launch-invalidating finding): rows were
    ONLY emitted at --ckpt-every cadence, so a timeout-stop smoke killed by
    SIGTERM before the first checkpoint-due epoch destroyed every regime
    timing row in the in-memory buffer (r5 masked this — it never reached
    epoch 1). The stdout mirror is captured by the provider lane log (harvested
    even at rc=124, proven in-vivo by r5), while the on-disk trajectory JSONL
    keeps its checkpoint-consistent resume invariant unchanged. Mirror rows
    carry stdout_mirror=true so harvest dedupes them against flushed rows.
    """
    print(json.dumps({**row, "stdout_mirror": True}), flush=True)
    rows.append(row)


def _checkpoint_blob(
    model,
    ema,
    optimizer,
    epoch: int,
    config_hash: str,
    argv: tuple[str, ...],
    *,
    pair_cursor: DeterministicPairCursor | None = None,
    controller_state: Mapping[str, Any] | None = None,
    protected_seed=None,
    seed_optimizer=None,
    tail_controller: TailController | None = None,
    scorer_sha256: Mapping[str, str] | None = None,
) -> dict:
    import torch

    return {
        "schema": "v9_cgauge_torch_resume_v2",
        "epoch": int(epoch),
        "model": model.state_dict(),
        "ema": ema.state_dict(),  # inference/deploy authority is the EMA shadow
        "optimizer": optimizer.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
        "config_hash": config_hash,
        "dsl_argv": list(argv),
        "scorer_sha256": (
            None
            if scorer_sha256 is None
            else {name: value.lower() for name, value in scorer_sha256.items()}
        ),
        "pair_cursor": pair_cursor.state_dict() if pair_cursor is not None else None,
        "controller_state": copy.deepcopy(dict(controller_state or {})),
        "protected_seed": (
            protected_seed.state_dict() if protected_seed is not None else None
        ),
        "seed_optimizer": (
            seed_optimizer.state_dict() if seed_optimizer is not None else None
        ),
        "tail_controller": (
            tail_controller.state_dict() if tail_controller is not None else None
        ),
    }


def _restore(
    blob,
    model,
    ema,
    optimizer,
    expected_hash: str,
    *,
    pair_cursor: DeterministicPairCursor | None = None,
    controller_state: dict[str, Any] | None = None,
    protected_seed=None,
    seed_optimizer=None,
    tail_controller: TailController | None = None,
) -> int:
    import torch

    if blob.get("config_hash") != expected_hash:
        raise ValueError("resume config hash differs from the typed V9 CGauge program; refusing drift")
    model.load_state_dict(blob["model"], strict=True)
    ema.load_state_dict(blob["ema"], strict=True)
    optimizer.load_state_dict(blob["optimizer"])
    torch.set_rng_state(blob["torch_rng"])
    if torch.cuda.is_available() and blob.get("cuda_rng"):
        torch.cuda.set_rng_state_all(blob["cuda_rng"])
    np.random.set_state(blob["numpy_rng"])
    random.setstate(blob["python_rng"])
    if pair_cursor is not None and blob.get("pair_cursor") is not None:
        pair_cursor.load_state_dict(blob["pair_cursor"])
    if controller_state is not None:
        controller_state.clear()
        controller_state.update(copy.deepcopy(blob.get("controller_state", {})))
    seed_blob = blob.get("protected_seed")
    if protected_seed is not None:
        if seed_blob is None:
            raise ValueError("active typed protected seed is missing from resume checkpoint")
        protected_seed.load_state_dict(seed_blob)
    elif seed_blob is not None:
        raise ValueError("resume checkpoint contains a protected seed but typed config disabled it")
    seed_opt_blob = blob.get("seed_optimizer")
    if seed_optimizer is not None:
        if seed_opt_blob is None:
            raise ValueError("active typed seed optimizer is missing from resume checkpoint")
        seed_optimizer.load_state_dict(seed_opt_blob)
    elif seed_opt_blob is not None:
        raise ValueError("resume checkpoint contains a seed optimizer but typed config disabled it")
    tail_blob = blob.get("tail_controller")
    if tail_controller is not None:
        if tail_blob is None:
            # Legacy pre-tail checkpoint is valid only before the first cycle.
            if int(blob["epoch"]) >= int(tail_controller.cycle_start_ep or 10**18):
                raise ValueError("active tail cycle is missing from resume checkpoint")
        else:
            tail_controller.load_state_dict(tail_blob)
    elif tail_blob is not None:
        raise ValueError("resume checkpoint contains tail state but typed config disabled it")
    return int(blob["epoch"])


def _softmax_temp_at_epoch(epoch: int, total_epochs: int, flags: Mapping[str, Any]) -> float:
    """Exact pure twin of the MLX temperature continuation schedule."""
    anneal = int(flags.get("--anneal-epochs", 0) or total_epochs)
    progress = (int(epoch) - 1) / max(anneal - 1, 1)
    start = float(flags["--softmax-temp-start"])
    end = float(flags["--softmax-temp-end"])
    shape = str(flags.get("--tau-anneal-shape", "cosine"))
    if shape == "geometric":
        return float(start * (end / start) ** progress)
    if shape == "cosine_hold":
        hold = float(flags.get("--tau-hold-frac", 1.0))
        if hold < 1.0:
            if progress >= hold:
                return end
            progress /= hold
    return float(end + 0.5 * (start - end) * (1.0 + np.cos(np.pi * progress)))


def _hosc_beta_at_epoch(epoch: int, total_epochs: int, flags: Mapping[str, Any]) -> float:
    """Exact pure twin of the MLX HOSC sharpening schedule."""
    start = float(flags["--hosc-beta"])
    end = float(flags.get("--hosc-beta-end", start))
    if end == start:
        return start
    anneal = int(flags.get("--anneal-epochs", 0) or total_epochs)
    progress = (int(epoch) - 1) / max(anneal - 1, 1)
    if str(flags.get("--hosc-beta-anneal", "linear")) == "cosine":
        return float(end + 0.5 * (start - end) * (1.0 + np.cos(np.pi * progress)))
    return float(start + (end - start) * progress)


def _ema_update(ema, model, decay: float) -> None:
    with __import__("torch").no_grad():
        for dst, src in zip(ema.parameters(), model.parameters(), strict=True):
            dst.mul_(decay).add_(src, alpha=1.0 - decay)
        for dst, src in zip(ema.buffers(), model.buffers(), strict=True):
            dst.copy_(src)


def _load_scorers(device):
    from modules import DistortionNet, posenet_sd_path, segnet_sd_path

    from tac.boundary_math.seg_core import load_real_segnet

    seg = load_real_segnet(str(device)).eval()
    dn = DistortionNet().eval()
    dn.load_state_dicts(posenet_sd_path, segnet_sd_path, device)
    pose = dn.posenet.to(device).eval()
    for net in (seg, pose):
        for p in net.parameters():
            p.requires_grad_(False)
    return seg, pose


def _pose6(pose_net, frames_nhwc):
    # frames: (B,2,H,W,3), scorer preprocess is part of the real path.
    x = frames_nhwc.permute(0, 1, 4, 2, 3).contiguous()
    out = pose_net(pose_net.preprocess_input(x))
    pose = out["pose"] if isinstance(out, dict) else out
    half = next((h.out // 2 for h in pose_net.hydra.heads if h.name == "pose"), pose.shape[-1] // 2)
    return pose[:, :half]


def _required_typed_flags(flags: Mapping[str, Any], names: tuple[str, ...]) -> None:
    missing = [name for name in names if name not in flags]
    if missing:
        raise ValueError(
            "active V9 mechanism lacks typed DSL companion values: " + ", ".join(missing)
        )


def _attach_generated_pose_carrier(model, flags, gt_poses, native_hw, device):
    """Attach the typed generated/table pose carrier before EMA and optimizer creation."""
    if not bool(flags.get("--pose-carrier", False)):
        return None, {"active": False}
    needed = (
        "--pose-carrier-source", "--pose-carrier-residual-mode",
        "--pose-carrier-residual-scale", "--pose-carrier-s-t",
        "--pose-carrier-s-r", "--pose-carrier-pitch",
    )
    _required_typed_flags(flags, needed)
    if flags["--pose-carrier-source"] != "generated":
        raise ValueError("Torch V9 carrier currently requires typed source=generated")
    if flags["--pose-carrier-residual-mode"] != "table":
        raise ValueError("Torch V9 carrier currently requires typed residual-mode=table")
    from tac.boundary_math.warp_real_luma_frame0 import (
        GroundHomographyGeom,
        xi_from_pose_calibration,
    )

    native_hw = tuple(int(x) for x in native_hw)
    geom = GroundHomographyGeom.eon(
        native_hw=native_hw, pitch=float(flags["--pose-carrier-pitch"])
    )
    xi_stored = np.stack([
        xi_from_pose_calibration(
            np.asarray(pose),
            float(flags["--pose-carrier-s-t"]),
            float(flags["--pose-carrier-s-r"]),
            float(flags["--pose-carrier-pitch"]),
        )
        for pose in np.asarray(gt_poses)
    ]).astype(np.float32)
    carrier = TorchPoseCarrier.build(
        xi_stored,
        geom,
        residual_scale=float(flags["--pose-carrier-residual-scale"]),
    ).to(device)
    model.pose_carrier = carrier
    return carrier, {
        "active": True,
        "source": "generated",
        "residual_mode": "table",
        "native_hw": list(native_hw),
        "s_t": float(flags["--pose-carrier-s-t"]),
        "s_r": float(flags["--pose-carrier-s-r"]),
        "pitch": float(flags["--pose-carrier-pitch"]),
        "n_pairs": len(gt_poses),
    }


def _run_structured_prefit(model, flags, lstars, feats, *, seed: int, is_resume: bool):
    """Run the active typed structured-core prefit only on a fresh model."""
    if not bool(flags.get("--structured-init", False)):
        return {"active": False, "applied": False}
    needed = (
        "--structured-init-include-lane", "--structured-init-thresh",
        "--structured-init-steps", "--structured-init-lr",
        "--structured-init-subsample", "--structured-init-sdf-clip",
    )
    _required_typed_flags(flags, needed)
    if is_resume:
        return {"active": True, "applied": False, "reason": "resume_preserves_checkpoint"}
    cfg = model.cfg
    if tuple(np.asarray(lstars).shape[1:]) != (cfg.render_h, cfg.render_w):
        raise ValueError("structured-init L* shape differs from the typed render geometry")
    phi_hwk, roles, meta = build_static_core_phi_target(
        np.asarray(lstars),
        n_classes=cfg.n_classes,
        include_lane=bool(flags["--structured-init-include-lane"]),
        static_thresh=float(flags["--structured-init-thresh"]),
    )
    clip = float(flags["--structured-init-sdf-clip"])
    target = np.clip(phi_hwk.reshape(-1, cfg.n_classes), -clip, clip).astype(np.float32)
    import torch

    target_t = torch.as_tensor(target, device=feats.device)
    row = structured_sdf_prefit(
        model,
        feats,
        target_t,
        steps=int(flags["--structured-init-steps"]),
        lr=float(flags["--structured-init-lr"]),
        subsample=int(flags["--structured-init-subsample"]),
        seed=int(seed),
    )
    with torch.no_grad():
        _rgb, pred = model(feats, torch.zeros(1, dtype=torch.long, device=feats.device))
    disagree = float(np.mean(
        pred[0].argmax(-1).detach().cpu().numpy() != target.argmax(-1)
    ))
    return {
        "active": True, "applied": True, **row,
        "direct_argmax_disagree": disagree,
        "roles": roles.as_dict(),
        **{k: v for k, v in meta.items() if k != "roles"},
    }


def _generated_pose_pair_dispatch(
    model,
    feats,
    pair_indices,
    pose_carrier,
    cfg,
    *,
    lane_band=None,
    protected_seed=None,
    seed_weight: float = 1.0,
    return_probe_inputs: bool = False,
    return_witness_alone: bool = False,
    r_operator=contest_r,
):
    """MLX-authority generated/table dispatch: plain f0 up->warp->R-down, f1 witness R.

    ``pair_indices`` indexes pairs, not frame codes. The attached carrier is the only
    consumer of its trainable dxi. Frame1 never reads dxi, preserving the SegNet-free
    frame0 / scorer-visible frame1 separation.
    """
    import torch
    import torch.nn.functional as F

    pair_indices = torch.as_tensor(pair_indices, device=feats.device, dtype=torch.long)
    code0 = 2 * pair_indices
    code1 = code0 + 1
    raw0, _phi0 = model(feats, code0)
    if lane_band is None:
        raw1, phi1 = model(feats, code1)
    else:
        raw1, phi1, witness_margin, lane_rgb = model.lane_band_fields(
            feats, code1, lane_cls=int(lane_band["lane_cls"])
        )
        if "coverage" in lane_band:
            coverage = lane_band["coverage"].index_select(0, pair_indices).to(raw1.dtype)
        else:
            coverage = torch.stack(
                [
                    torch.as_tensor(
                        lane_band["priors"][int(index)].coverage,
                        device=raw1.device,
                        dtype=raw1.dtype,
                    ).reshape(-1)
                    for index in pair_indices.detach().cpu().tolist()
                ]
            )
        uncertainty = (
            (float(lane_band["tau"]) - witness_margin)
            / max(float(lane_band["eps"]), 1e-6)
            + 0.5
        ).clamp(0.0, 1.0).detach()
        alpha = (coverage * float(lane_band["weight"]) * uncertainty)[..., None]
        raw1 = raw1 * (1.0 - alpha) + lane_rgb * alpha
    n = pair_indices.numel()
    raw0 = raw0.reshape(n, cfg.render_h, cfg.render_w, 3)
    raw1 = raw1.reshape(n, cfg.render_h, cfg.render_w, 3)
    witness_alone_raw1 = raw1
    if protected_seed is not None:
        raw1 = protected_seed.compose(raw1, pair_indices, weight=float(seed_weight))
    native0 = F.interpolate(
        raw0.permute(0, 3, 1, 2),
        size=(cfg.camera_h, cfg.camera_w),
        mode="bicubic",
        align_corners=False,
    ).permute(0, 2, 3, 1)
    native0 = torch.clamp(round_ste(native0), 0.0, 255.0)
    warped0 = pose_carrier(native0, pair_indices)
    scored0 = F.interpolate(
        warped0.permute(0, 3, 1, 2),
        size=(cfg.render_h, cfg.render_w),
        mode="bilinear",
        align_corners=False,
    ).permute(0, 2, 3, 1).contiguous()
    scored1 = r_operator(raw1, output_hw=(cfg.render_h, cfg.render_w))
    frames = torch.stack((scored0, scored1), dim=1)
    phi = phi1.reshape(n, cfg.render_h, cfg.render_w, cfg.n_classes)
    witness_alone = None
    if return_witness_alone:
        witness_alone1 = r_operator(
            witness_alone_raw1, output_hw=(cfg.render_h, cfg.render_w)
        )
        witness_alone = torch.stack((scored0, witness_alone1), dim=1)
    if return_probe_inputs:
        result = (frames, phi, {
            "native0": native0.detach(),
            "scored1": scored1.detach(),
        })
        return (*result, witness_alone) if return_witness_alone else result
    if return_witness_alone:
        return frames, phi, witness_alone
    return frames, phi


def _jacobian_probe_pair_indices(
    flags: Mapping[str, Any], gt_poses: np.ndarray, epoch: int
) -> list[int]:
    """Typed cadence plus deterministic |ego-t|-stratified probe selection."""
    if not bool(flags.get("--jacobian-basin-telemetry", False)):
        return []
    eval_every = max(1, int(flags["--eval-every"]))
    basin_every = max(1, int(flags["--jacobian-basin-every"]))
    if int(epoch) % (eval_every * basin_every) != 0:
        return []
    if not bool(flags["--jacobian-basin-stratify-t"]):
        raise ValueError("active V9 Torch J_xi probe requires typed motion stratification")
    from tac.witness_control.jacobian_basin import (
        motion_magnitude,
        stratified_indices_by_motion,
    )

    motions = [motion_magnitude(pose6) for pose6 in np.asarray(gt_poses)]
    return stratified_indices_by_motion(
        motions, int(flags["--jacobian-basin-k-pairs"])
    )


def _polyak_checkpoint_blob(
    controller: TorchV9ControllerRuntime,
    epoch: int,
    config_hash: str,
    argv: tuple[str, ...],
    scorer_sha256: Mapping[str, str] | None = None,
) -> dict[str, Any] | None:
    """Materialize the additional Polyak candidate without replacing EMA."""
    import torch

    candidate = controller.polyak_candidate()
    if candidate is None:
        return None
    return {
        "schema": "v9_cgauge_torch_polyak_v1",
        "epoch": int(epoch),
        "polyak": {
            key: torch.as_tensor(value).detach().cpu().clone()
            for key, value in candidate.items()
        },
        "count": int(controller.polyak.count),
        "config_hash": config_hash,
        "dsl_argv": list(argv),
        "scorer_sha256": (
            None
            if scorer_sha256 is None
            else {name: value.lower() for name, value in scorer_sha256.items()}
        ),
        "authority": "[contest-CUDA training-advisory] NON-PROMOTABLE",
    }


def _accumulated_pair_step(
    model,
    optimizer,
    pair_indices,
    loss_builder,
    *,
    grad_clip: float,
):
    """Mean a complete pair chunk, then atomically accept or reject one update.

    This mirrors the MLX authority: every pair in the chunk is evaluated exactly
    once, the mean loss is formed before backward, and the finite/spike guard is
    applied to the chunk as a whole.  Telemetry therefore counts accepted chunks,
    not accepted members within a partially retained chunk.
    """
    import torch

    optimizer.zero_grad(set_to_none=True)
    losses = []
    for pair_index in pair_indices:
        loss = loss_builder(int(pair_index))
        if loss is None:
            optimizer.zero_grad(set_to_none=True)
            return {
                "weights_stepped": False, "accepted": 0, "attempted": 1,
                "accepted_frac": 0.0, "loss_mean": None, "group_norms": {},
                "pair_count": len(pair_indices),
            }
        losses.append(loss)
    if not losses:
        optimizer.zero_grad(set_to_none=True)
        return {
            "weights_stepped": False, "accepted": 0, "attempted": 0,
            "accepted_frac": 0.0, "loss_mean": None, "group_norms": {},
            "pair_count": 0,
        }
    differentiable_mean = torch.stack(losses).mean()
    loss_mean = differentiable_mean.detach()
    if not bool(torch.isfinite(loss_mean)):
        optimizer.zero_grad(set_to_none=True)
        return {
            "weights_stepped": False, "accepted": 0, "attempted": 1,
            "accepted_frac": 0.0, "loss_mean": loss_mean, "group_norms": {},
            "pair_count": len(losses),
        }
    differentiable_mean.backward()
    group_norms = clip_grad_groups(parameter_groups(model), float(grad_clip))
    optimizer.step()
    return {
        "weights_stepped": True, "accepted": 1, "attempted": 1,
        "accepted_frac": 1.0, "loss_mean": loss_mean, "group_norms": group_norms,
        "pair_count": len(losses),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.num_pairs <= 0:
        raise ValueError("num-pairs must be positive")
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    if args.stop_after_epochs is not None and not 1 <= args.stop_after_epochs <= 3:
        raise ValueError("stop-after-epochs must be in 1..3")
    if args.device not in {"cuda", "cpu"}:
        raise ValueError("device must be exactly 'cuda' or 'cpu'")
    if args.preflight_only and args.verify_only:
        raise ValueError("preflight-only and verify-only are mutually exclusive")
    scorer_expected_values = {
        "segnet": args.expected_segnet_sha256,
        "posenet": args.expected_posenet_sha256,
    }
    supplied_scorer_expected = {
        name: value for name, value in scorer_expected_values.items() if value is not None
    }
    if supplied_scorer_expected and len(supplied_scorer_expected) != 2:
        raise ValueError("expected scorer SHA custody requires both segnet and posenet")
    for name, value in supplied_scorer_expected.items():
        if len(value) != 64 or any(ch not in "0123456789abcdefABCDEF" for ch in value):
            raise ValueError(f"expected {name} SHA-256 must be exactly 64 hex characters")
    if (args.preflight_only or args.no_implicit_resume) and not supplied_scorer_expected:
        raise ValueError(
            "strict/preflight execution requires expected SegNet and PoseNet SHA-256 custody"
        )
    out = Path(args.out_dir)
    if any(str(out.resolve()).startswith(x) for x in _FORBIDDEN_TMP):
        raise ValueError("out-dir is a tmp-class path; use the SSD/repo tier")

    flags, cfg_hash, dsl_argv = derive_config(args)
    resume_path = (
        None
        if args.verify_only
        else _resolve_resume_intent(
            out,
            args.resume_from,
            no_implicit_resume=args.no_implicit_resume,
        )
    )
    gt = None if args.verify_only else _load_validated_gt_cache(args.gt_cache, args.num_pairs)
    resume_blob = (
        _load_validated_resume(
            resume_path,
            cfg_hash,
            args.epochs,
            expected_scorer_sha256=supplied_scorer_expected,
            require_scorer_custody=(args.preflight_only or args.no_implicit_resume),
        )
        if resume_path is not None
        else None
    )
    planned_start = int(resume_blob["epoch"]) if resume_blob is not None else 0
    run_first_epoch, run_end_epoch = _runtime_epoch_window(
        planned_start, args.epochs, args.stop_after_epochs
    )
    gt_geometry = None if gt is None else _validate_gt_geometry(gt, flags)
    scorer_custody = (
        None
        if args.verify_only
        else _validate_scorer_custody(expected_sha256=supplied_scorer_expected)
    )
    scorer_sha256 = (
        None
        if scorer_custody is None
        else {name: row["sha256"] for name, row in scorer_custody.items()}
    )
    coverage = cuda_v9_port_receipt()
    if args.preflight_only and coverage["status"] != "COMPLETE_1_TO_1":
        raise RuntimeError(
            "NO-FAKE REFUSAL: CPU preflight found incomplete V9 CUDA control semantics; "
            f"unclosed surfaces: {coverage['blockers']}"
        )
    if args.preflight_only:
        assert gt is not None and gt_geometry is not None and scorer_custody is not None
        import torch

        preflight_seg, preflight_pose = _load_scorers(torch.device("cpu"))
        preflight_scorers = {"segnet": preflight_seg, "posenet": preflight_pose}
        scorer_load_receipt = {
            name: {
                "class": type(network).__name__,
                "eval": not bool(network.training),
                "frozen": not any(parameter.requires_grad for parameter in network.parameters()),
            }
            for name, network in preflight_scorers.items()
        }
        if not all(
            row["eval"] and row["frozen"] for row in scorer_load_receipt.values()
        ):
            raise RuntimeError("CPU scorer constructor/load did not yield eval frozen networks")
        del preflight_seg, preflight_pose, preflight_scorers
        print(
            json.dumps(
                {
                    "schema": "v9_cgauge_torch_preflight.v1",
                    "status": "passed",
                    "dsl_config": args.dsl_config,
                    "config_hash": cfg_hash,
                    "typed_total_epochs": args.epochs,
                    "runtime_stop_after_epochs": args.stop_after_epochs,
                    "runtime_epoch_window": [run_first_epoch, run_end_epoch],
                    "num_pairs": args.num_pairs,
                    "gt_cache": str(Path(args.gt_cache)),
                    "gt_geometry": gt_geometry,
                    "scorer_custody": scorer_custody,
                    "scorer_constructor_load": {
                        "status": "passed",
                        "device": "cpu",
                        "networks": scorer_load_receipt,
                    },
                    "cuda_v9_port_coverage": {
                        "status": coverage["status"],
                        "blockers": coverage["blockers"],
                    },
                    "resume_checkpoint": (
                        None if resume_path is None else str(resume_path)
                    ),
                    "resume_epoch": planned_start,
                    "no_implicit_resume": bool(args.no_implicit_resume),
                    "output_created": False,
                    "authority": "runtime-input-validation-only",
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    bank = CurveletBankConfig()
    B = curvelet_directional_B(bank, max_freq=float(flags["--max-bank-freq"]))
    coords = build_coords(int(flags["--render-h"]), int(flags["--render-w"]))
    feats_np = curvelet_feats(coords, B)
    cfg = CudaLevelSetConfig(
        n_pairs=args.num_pairs, in_feat=feats_np.shape[1], hidden_dim=int(flags["--hidden-dim"]),
        n_hidden=int(flags["--n-hidden"]), mod_dim=int(flags["--mod-dim"]),
        activation=str(flags["--activation"]), hosc_beta=float(flags["--hosc-beta"]),
        hosc_omega=float(flags["--hosc-omega"]), softmax_temp=float(flags["--softmax-temp-start"]),
        chroma=bool(flags.get("--chroma", False)), render_h=int(flags["--render-h"]),
        render_w=int(flags["--render-w"]),
    )
    if gt is not None and gt_geometry != {
        "render_hw": [cfg.render_h, cfg.render_w],
        "camera_hw": [cfg.camera_h, cfg.camera_w],
    }:
        raise RuntimeError("validated GT geometry drifted during CUDA config construction")

    import torch
    if args.device == "cuda" and not torch.cuda.is_available() and not args.verify_only:
        raise RuntimeError("CUDA requested but unavailable (fail closed; use --verify-only for $0 local proof)")
    device = torch.device("cpu" if args.verify_only else args.device)
    execution_policy = select_torch_execution_policy(
        device, compile_mode=args.torch_compile_mode
    )
    if device.type == "cuda":
        apply_torch_execution_policy(execution_policy)
    else:
        torch.use_deterministic_algorithms(True)
    torch.manual_seed(int(flags["--seed"]))
    np.random.seed(int(flags["--seed"]))
    random.seed(int(flags["--seed"]))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(flags["--seed"]))
    model = TorchLevelSetWitness.build(cfg, seed=int(flags["--seed"])).to(device)
    parity = forward_parity_against_numpy(model, feats_np[: min(257, len(feats_np))])
    row = {"stage": "cuda_numpy_forward_parity", "backend": str(device), **parity,
           "measured": True, "promotion_eligible": False}
    print(json.dumps(row), flush=True)
    print(json.dumps({"stage": "cuda_v9_port_coverage", **coverage}), flush=True)
    # F7 provenance: NUMPY_FP32_PARITY_COSINE_BAR = the CLAUDE.md
    # deterministic-reproducibility item (3) numpy-fp32 authority parity bar.
    parity_passed = bool(
        parity["argmax_equal"]
        and parity["cosine_phi"] >= NUMPY_FP32_PARITY_COSINE_BAR
    )
    if not parity_passed and not args.verify_only:
        raise RuntimeError("CUDA/NumPy forward parity gate failed; refusing before compile or scorers")

    compile_probe_result: dict[str, Any] | None = None
    if args.compile_probe or device.type == "cuda":
        f = torch.as_tensor(feats_np[:64], device=device)
        ci = torch.tensor([0], device=device)
        compile_probe_result = compile_identity_probe(
            model, f, ci, lambda rgb, phi: rgb.square().mean() + phi.square().mean(),
            # Probe the SAME Inductor mode the training region will adopt so the
            # r3 adoption rule validates the actual artifact (policy None =>
            # compile disabled; probe still runs cheaply in 'default').
            mode=execution_policy.compile_mode or "default",
        )
        print(json.dumps({"stage": "backend_fp_reorder_probe", "backend": str(device),
                          **compile_probe_result}), flush=True)
        if not compile_probe_result.get("adoptable", False):
            raise RuntimeError(
                "compile probe is non-adoptable; refusing before structured prefit/scorers"
            )

    if args.verify_only:
        return 0 if parity_passed else 2

    if coverage["status"] != "COMPLETE_1_TO_1":
        raise RuntimeError(
            "NO-FAKE REFUSAL: active V9 CUDA control semantics are not 1:1; "
            f"unclosed surfaces: {coverage['blockers']}"
        )
    seg, pose = _load_scorers(device)

    assert gt is not None
    lstars = gt["lstars"]
    margins = gt["margins"]
    gt_f1 = gt["gt_f1"]
    gt_poses = gt["gt_poses"]
    if bool(flags.get("--dseg-aware-taper", False)):
        from tac.boundary_math.dseg_aware_fourier_taper import (
            apply_dseg_aware_fourier_taper,
            compute_dseg_aware_fourier_taper,
            saliency_from_margins,
        )

        taper_scale = float(flags["--dseg-aware-taper-scale"])
        taper_saliency = saliency_from_margins(
            margins,
            scale=None if taper_scale <= 0.0 else taper_scale,
            target_hw=(cfg.render_h, cfg.render_w),
        )
        taper = compute_dseg_aware_fourier_taper(
            feats_np,
            taper_saliency,
            strength=float(flags["--dseg-aware-taper-strength"]),
            floor=float(flags["--dseg-aware-taper-floor"]),
        )
        feats_np = apply_dseg_aware_fourier_taper(feats_np, taper).astype(np.float32)
        print(
            json.dumps(
                {
                    "stage": "dseg_aware_taper",
                    "n_cols": int(taper.shape[0]),
                    "strength": float(flags["--dseg-aware-taper-strength"]),
                    "scale": "auto" if taper_scale <= 0.0 else round(taper_scale, 6),
                    "taper_min": round(float(taper.min()), 4),
                    "taper_max": round(float(taper.max()), 4),
                    "taper_mean": round(float(taper.mean()), 4),
                    "note": (
                        "#121 byte-neutral spectral reallocation by GT margin saliency; "
                        "RE-VALIDATE at convergence; training-advisory NON-PROMOTABLE"
                    ),
                }
            ),
            flush=True,
        )
    # Generated source never reads/materializes gt_f0. Camera geometry is shared
    # by the already-required gt_f1 array and the canonical receiver dimensions.
    native_hw = tuple(int(x) for x in gt_f1.shape[1:3])
    if native_hw != (cfg.camera_h, cfg.camera_w):
        raise ValueError(
            f"GT camera geometry {native_hw} differs from Torch receiver {(cfg.camera_h, cfg.camera_w)}"
        )
    feats = torch.as_tensor(feats_np, device=device)
    lstars_device = torch.as_tensor(lstars, device=device, dtype=torch.long)
    margins_device = torch.as_tensor(margins, device=device, dtype=torch.float32)
    gt_poses_device = torch.as_tensor(gt_poses, device=device, dtype=torch.float32)
    pose_carrier, pose_carrier_row = _attach_generated_pose_carrier(
        model, flags, gt_poses, native_hw, device
    )
    print(json.dumps({"stage": "pose_carrier", **pose_carrier_row}), flush=True)
    resume_will_load = resume_blob is not None
    if flags.get("--palette-anchor", False) and not resume_will_load:
        import torch.nn.functional as F

        sums = np.zeros((cfg.n_classes, 3), np.float64)
        cnts = np.zeros(cfg.n_classes, np.float64)
        for pi in range(min(args.num_pairs, 64)):
            frame = torch.from_numpy(np.asarray(gt_f1[pi], np.float32)).permute(2, 0, 1)[None]
            small = F.interpolate(
                frame, size=(cfg.render_h, cfg.render_w), mode="bilinear", align_corners=False
            )[0].permute(1, 2, 0).numpy()
            for cls in range(cfg.n_classes):
                mask = lstars[pi] == cls
                if mask.any():
                    sums[cls] += small[mask].sum(0)
                    cnts[cls] += int(mask.sum())
        mean = np.where(cnts[:, None] > 0, sums / np.maximum(cnts[:, None], 1), 127.0)
        clipped = np.clip(mean / 255.0, 1e-3, 1.0 - 1e-3)
        palette = np.log(clipped / (1.0 - clipped)).astype(np.float32)
        with torch.no_grad():
            model.palette.copy_(torch.as_tensor(palette, device=device))
    structured_row = _run_structured_prefit(
        model,
        flags,
        lstars,
        feats,
        seed=int(flags["--seed"]),
        is_resume=resume_will_load,
    )
    print(json.dumps({"stage": "structured_init", **structured_row}), flush=True)
    counts = np.bincount(lstars.reshape(-1), minlength=cfg.n_classes).astype(np.float64)
    priors = counts / counts.sum()
    la_tau = float(flags.get("--logit-adjust-loss-tau", 0.0))
    la_spec = str(flags.get("--logit-adjust-classes", "all"))
    la_allowed = (
        tuple(range(cfg.n_classes))
        if la_spec.lower() == "all"
        else tuple(int(x) for x in la_spec.split(",") if x.strip())
    )
    logit_offsets_np = np.zeros(cfg.n_classes, np.float32)
    if la_tau != 0.0:
        raw = la_tau * np.log(np.maximum(priors, 1e-8))
        for cls in la_allowed:
            logit_offsets_np[cls] = raw[cls]
    persist_classes = tuple(
        int(x) for x in str(flags.get("--persistence-classes", "3")).split(",")
        if x.strip() and x.strip().lower() != "auto"
    )
    from tac.boundary_math.island_protection import identify_island_classes
    island_detection = identify_island_classes(lstars, n_classes=cfg.n_classes)
    island_targets = TorchIslandTargetRuntime(
        lstars,
        lane_cls=island_detection.lane_cls,
        movable_cls=island_detection.movable_cls,
        flags=flags,
        device=device,
    )
    protected_seed = (
        island_targets.build_protected_seed(gt_f1)
        if bool(flags.get("--seed-islands", False))
        else None
    )
    from tac.canonical_equations.chan_vese_area_constraint_birth_balance_20260708 import (
        area_constraint_lambda,
    )
    area_classes = tuple(
        int(x) for x in str(flags.get("--area-constraint-classes", "1,3")).split(",") if x.strip()
    )
    area_lambdas = {
        cls: area_constraint_lambda(
            float(priors[cls]),
            birth_force=float(flags.get("--area-constraint-birth-force", 1.0)),
            tolerance=float(flags.get("--area-constraint-tolerance", 0.25)),
        )
        for cls in area_classes
    } if flags.get("--area-constraint-birth", False) else {}
    lane_band = None
    if flags.get("--lane-render-band", False):
        from tac.boundary_math.dash_comb import build_combed_lane_band_priors

        lane_priors, lane_comb_fit = build_combed_lane_band_priors(
            lstars,
            gt_poses,
            lane_cls=island_detection.lane_cls,
            softness=float(flags.get("--lane-band-softness", 1.0)),
            dash_forward_max_m=float(flags.get("--lane-band-dash-forward-max-m", 55.0)),
            comb_softness_m=float(flags.get("--lane-band-comb-softness-m", 0.3)),
        )
        lane_band = {
            "priors": lane_priors,
            "coverage": torch.as_tensor(
                np.stack([lane_priors[index].coverage.reshape(-1) for index in range(args.num_pairs)]),
                device=device,
            ),
            "lane_cls": int(island_detection.lane_cls),
            "tau": float(flags.get("--lane-band-tau", 0.85)),
            "eps": float(flags.get("--lane-band-eps", 0.35)),
            "weight": float(flags.get("--lane-band-weight", 1.0)),
            "start_epoch": int(flags.get("--lane-band-start-epoch", 500)),
            "comb": {
                "period_m": float(lane_comb_fit.period_m),
                "duty": float(lane_comb_fit.duty),
                "ego_scale": float(lane_comb_fit.scale),
            },
        }
    phase_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    phase_weight = float(flags.get("--seg-phase-advect-weight", 0.0))
    phase_start = int(flags.get("--seg-phase-advect-start-epoch", 0))
    if phase_weight > 0.0:
        from tac.boundary_math.phase_primitives import (
            advect_tie_field_numpy,
            cross_scored_frame_xi_interp,
            gt_tie_targets_numpy,
        )
        from tac.boundary_math.warp_real_luma_frame0 import (
            GroundHomographyGeom,
            xi_from_pose_calibration,
        )

        phase_geom = GroundHomographyGeom.eon(
            native_hw=(cfg.render_h, cfg.render_w),
            pitch=float(flags.get("--gfc-pitch", -0.01)),
        )
        phase_xi = [
            xi_from_pose_calibration(
                gt_poses[pi],
                float(flags.get("--gfc-s-t", -0.003224707899359239)),
                float(flags.get("--gfc-s-r", 0.0)),
                float(flags.get("--gfc-pitch", -0.01)),
            )
            for pi in range(args.num_pairs)
        ]
        phase_classes = {
            int(x) for x in str(flags.get("--seg-phase-advect-classes", "0,1,2")).split(",")
            if x.strip()
        }
        phase_band = float(flags.get("--seg-phase-advect-band", 2.0))

        def phase_provider(pair_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            if pair_index in phase_cache:
                return phase_cache[pair_index]
            _t, direction, _active = gt_tie_targets_numpy(
                lstars[pair_index], margins[pair_index], band=phase_band
            )
            if pair_index == 0:
                ref = np.full_like(direction, -1.0, dtype=np.float32)
                weight = np.zeros_like(direction, dtype=np.float32)
            else:
                prev_t, _prev_dir, prev_active = gt_tie_targets_numpy(
                    lstars[pair_index - 1], margins[pair_index - 1], band=phase_band
                )
                xi_cross = cross_scored_frame_xi_interp(
                    phase_xi[pair_index - 1], phase_xi[pair_index]
                )
                ref_warp = advect_tie_field_numpy(
                    np.where(prev_t >= 0.0, prev_t, 0.0).astype(np.float32),
                    xi_cross,
                    phase_geom,
                )
                active_warp = advect_tie_field_numpy(
                    prev_active.astype(np.float32), xi_cross, phase_geom
                ) >= 0.5
                ref = np.where(active_warp, ref_warp, -1.0).astype(np.float32)
                weight = (
                    (margins[pair_index] < phase_band)
                    & np.isin(lstars[pair_index], list(phase_classes))
                    & active_warp
                ).astype(np.float32)
            phase_cache[pair_index] = (ref, direction.astype(np.float32), weight)
            return phase_cache[pair_index]
    temporal_weight = float(flags.get("--seg-temporal-screw-weight", 0.0))
    temporal_start = int(flags.get("--seg-temporal-screw-start-epoch", 0))
    temporal_grid_cache: dict[int, tuple[Any, Any]] = {}
    temporal_xi: list[np.ndarray] = []
    temporal_geom = None
    temporal_classes = tuple(
        int(x) for x in str(flags.get("--seg-temporal-screw-classes", "0,1,2")).split(",")
        if x.strip()
    )
    if temporal_weight > 0.0:
        from tac.boundary_math.warp_real_luma_frame0 import (
            GroundHomographyGeom,
            xi_from_pose_calibration,
        )

        temporal_geom = GroundHomographyGeom.eon(
            native_hw=(cfg.render_h, cfg.render_w),
            pitch=float(flags.get("--gfc-pitch", -0.01)),
        )
        temporal_xi = [
            xi_from_pose_calibration(
                gt_poses[pair_index],
                float(flags.get("--gfc-s-t", -0.003224707899359239)),
                float(flags.get("--gfc-s-r", 0.0)),
                float(flags.get("--gfc-pitch", -0.01)),
            )
            for pair_index in range(args.num_pairs)
        ]
    gt_f1_render_device = None
    if float(flags.get("--seg-chroma-boundary-weight", 0.0)) > 0.0:
        chroma_storage_dtype = (
            torch.bfloat16
            if device.type == "cuda" and execution_policy.amp_dtype == "bfloat16"
            else torch.float16 if device.type == "cuda" else torch.float32
        )
        resized_gt_chunks = []
        for chunk_start in range(0, args.num_pairs, int(flags["--accum-pairs"])):
            source = torch.from_numpy(
                np.asarray(
                    gt_f1[chunk_start : chunk_start + int(flags["--accum-pairs"])],
                    np.float32,
                )
            ).permute(0, 3, 1, 2)
            resized = torch.nn.functional.interpolate(
                source,
                size=(cfg.render_h, cfg.render_w),
                mode="bilinear",
                align_corners=False,
            ).permute(0, 2, 3, 1)
            resized_gt_chunks.append(resized.to(device=device, dtype=chroma_storage_dtype))
        gt_f1_render_device = torch.cat(resized_gt_chunks, dim=0)
    ema = copy.deepcopy(model).eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(flags["--lr"]),
                                  betas=(0.9, float(flags["--adam-beta2"])),
                                  weight_decay=float(flags["--weight-decay"]))
    if (
        resume_blob is not None
        and resume_blob.get("optimizer", {}).get("schema") == TorchMuonAdamW.SCHEMA
    ):
        muon_config = resume_blob["optimizer"].get("config", {})
        optimizer, _resume_muon_row = build_torch_muon_adamw(
            model,
            flags,
            total_epochs=args.epochs,
            start_epoch=int(muon_config["start_epoch"]),
        )
    seed_optimizer = (
        torch.optim.AdamW(
            protected_seed.parameters(),
            lr=float(flags["--seed-lr"]),
            weight_decay=0.0,
        )
        if protected_seed is not None
        else None
    )
    pair_cursor = DeterministicPairCursor(args.num_pairs, seed=int(flags["--seed"]))
    controller = TorchV9ControllerRuntime(
        flags,
        n_classes=cfg.n_classes,
        lane_cls=island_detection.lane_cls,
        movable_cls=island_detection.movable_cls,
    )
    tail_controller = None
    tail_start_epoch = 0
    if int(flags.get("--tail-cycles-max", 0)) > 0:
        if bool(flags.get("--tail-live-mq", False)):
            raise ValueError("typed live-mq tail is not available; V9 selects the halving fallback")
        tail_start_epoch = (
            int(flags["--tail-start-epoch"])
            if int(flags.get("--tail-start-epoch", 0)) > 0
            else int(flags["--muon-start-epoch"]) + int(flags["--tail-dwell-min"])
        )
        tail_tau0 = _softmax_temp_at_epoch(
            int(flags["--muon-start-epoch"]), args.epochs, flags
        )
        tail_controller = TailController(
            TailCycleConfig(
                k_max=int(flags["--tail-cycles-max"]),
                cycle_floor_epochs=float(flags["--tail-cycle-floor-epochs"]),
                dwell_min=int(flags["--tail-dwell-min"]),
                tau_halving=float(flags["--tail-tau-halving"]),
                tau_end=float(flags["--softmax-temp-end"]),
                lr_prop_coeff=float(flags["--tail-lr-prop-tau"]),
                stop_marginal_s=float(flags["--tail-stop-marginal-s"]),
            ),
            tau_ref=tail_tau0,
            lr_ref=float(flags["--muon-lr"]),
            tau0=tail_tau0,
        )
    controller_state: dict[str, Any] = {}
    start = 0
    if resume_will_load:
        assert resume_blob is not None
        start = _restore(
            resume_blob,
            model, ema, optimizer, cfg_hash,
            pair_cursor=pair_cursor, controller_state=controller_state,
            protected_seed=protected_seed, seed_optimizer=seed_optimizer,
            tail_controller=tail_controller,
        )
        if (
            tail_controller is not None
            and start >= tail_start_epoch
            and resume_blob.get("tail_controller") is None
        ):
            raise ValueError("post-tail resume checkpoint lacks governed tail-cycle state")
    if start != planned_start:
        raise RuntimeError("restored epoch differs from the pre-CUDA resume custody check")
    if run_first_epoch != start + 1:
        raise RuntimeError("runtime epoch window drifted after checkpoint restore")
    if controller_state:
        controller.load_state_dict(controller_state)
    controller_state = controller.state_dict()
    compiled_r_operator = contest_r
    loss_ops = {
        "pose": pose_objective_torch,
        "eikonal_length": eikonal_and_length,
        "realized_margin": realized_signed_margin,
        "island_birth": island_birth_from_signed_torch,
        "island_birth_perclass": island_birth_perclass_from_signed_torch,
        "persistence": persistence_topology_loss_torch,
        "area": area_constraint_torch,
        "weight_entropy": weight_entropy_rate_term_torch,
        "chroma": chroma_boundary_loss,
    }
    compile_receipts: dict[str, Any] = {}
    if device.type == "cuda":
        if compile_probe_result is None:
            raise RuntimeError("CUDA training requires the functional compile adoption probe")
        model._fields, model_compile_receipt = adopt_compiled_training_region(
            model._fields, execution_policy, compile_probe_result
        )
        seg.forward, seg_compile_receipt = adopt_compiled_training_region(
            seg.forward, execution_policy, compile_probe_result
        )
        pose.forward, pose_compile_receipt = adopt_compiled_training_region(
            pose.forward, execution_policy, compile_probe_result
        )
        compiled_r_operator, r_compile_receipt = adopt_compiled_training_region(
            contest_r, execution_policy, compile_probe_result
        )
        loss_compile_receipts = {}
        for loss_name, loss_fn in tuple(loss_ops.items()):
            loss_ops[loss_name], loss_receipt = adopt_compiled_training_region(
                loss_fn, execution_policy, compile_probe_result
            )
            loss_compile_receipts[loss_name] = loss_receipt.__dict__
        compile_receipts = {
            "film_hosc_inr": model_compile_receipt.__dict__,
            "frozen_segnet": seg_compile_receipt.__dict__,
            "frozen_posenet": pose_compile_receipt.__dict__,
            "contest_R": r_compile_receipt.__dict__,
            "loss_and_backward_regions": loss_compile_receipts,
            "cuda_graphs": "Inductor CUDA Graph Trees on fixed-shape compiled regions",
            "throughput": "UNMEASURED-pending-CUDA-dispatch",
        }
        amp_dtype = (
            torch.bfloat16
            if execution_policy.amp_dtype == "bfloat16"
            else torch.float16
        )
        torch.set_autocast_dtype("cuda", amp_dtype)
        torch.set_autocast_enabled("cuda", True)
    scaler = (
        torch.amp.GradScaler("cuda", enabled=bool(execution_policy.grad_scaler))
        if device.type == "cuda"
        else None
    )
    out.mkdir(parents=True, exist_ok=True)
    _atomic_json({"schema": "v9_cgauge_cuda_run_manifest_v1", "dsl_argv": list(dsl_argv),
                  "config_hash": cfg_hash, "device": str(device), "seed": int(flags["--seed"]),
                  "authority": "[contest-CUDA training-advisory] NON-PROMOTABLE",
                  "scorer_custody": scorer_custody,
                  "scorer_sha256": scorer_sha256,
                  "execution_policy": execution_policy.__dict__,
                  "controller": {
                      "schema": controller.SCHEMA,
                      "version": controller.VERSION,
                      "lane_cls": int(island_detection.lane_cls),
                      "movable_cls": int(island_detection.movable_cls),
                  },
                  "protected_seed": {
                      "active": protected_seed is not None,
                      "training_only": True,
                      "excluded_from_ema_and_deploy": True,
                  },
                  "compile_policy": "auto_adopt_after_functional_argmax_cosine_probe",
                  "compiled_training_regions": compile_receipts,
                  "fp_reorder_probe": compile_probe_result or {
                      "backend": str(device), "status": "UNMEASURED",
                  },
                  "runtime_epoch_plan": {
                      "start_exclusive": start,
                      "end_inclusive": run_end_epoch,
                      "stop_after_epochs": args.stop_after_epochs,
                      "typed_total_epochs": args.epochs,
                  },
                  "created_at_utc": _utc()}, out / TORCH_RUN_MANIFEST_JSON)

    ckpt_every = int(flags["--ckpt-every"])
    ema_decay = float(flags["--ema-decay"])
    base_logit_offsets = torch.as_tensor(logit_offsets_np, device=device)
    witness_alone_island = bool(flags.get("--witness-alone-island-loss", False))
    if witness_alone_island and protected_seed is None:
        raise ValueError(
            "typed witness-alone island loss requires the protected seed it excludes"
        )
    prev_stage = curriculum_stage(start, flags)
    t0 = time.time()
    last_completed_epoch = start
    tail_terminated = False
    pending_epoch_rows: list[dict[str, Any]] = []
    for epoch in range(run_first_epoch, run_end_epoch + 1):
        stage = curriculum_stage(epoch, flags)
        if stage != prev_stage:
            # Seal the COMPLETED prior stage before the first update of the new
            # stage.  This is both crash insurance and an independent A/B byte-
            # close surface; the filename encodes the last completed epoch.
            boundary_blob = _checkpoint_blob(
                model, ema, optimizer, epoch - 1, cfg_hash, dsl_argv,
                pair_cursor=pair_cursor, controller_state=controller.state_dict(),
                protected_seed=protected_seed, seed_optimizer=seed_optimizer,
                tail_controller=tail_controller, scorer_sha256=scorer_sha256,
            )
            _atomic_torch_save(
                boundary_blob,
                out / "stage_checkpoints" / f"ep{epoch - 1:05d}_{prev_stage}.pt",
            )
            boundary_polyak = _polyak_checkpoint_blob(
                controller, epoch - 1, cfg_hash, dsl_argv, scorer_sha256
            )
            if boundary_polyak is not None:
                _atomic_torch_save(
                    boundary_polyak,
                    out / "stage_checkpoints"
                    / f"ep{epoch - 1:05d}_{prev_stage}_polyak.pt",
                )
            prev_stage = stage
        model.train()
        with torch.no_grad():
            model.softmax_temp.fill_(_softmax_temp_at_epoch(epoch, args.epochs, flags))
            model.hosc_beta.fill_(_hosc_beta_at_epoch(epoch, args.epochs, flags))
        if pose_carrier is None:
            raise RuntimeError("active V9 generated pose carrier was not attached")
        controller_step = controller.begin_epoch(epoch)
        if controller_step.muon_start:
            if isinstance(optimizer, TorchMuonAdamW):
                raise RuntimeError("Muon event fired twice against an already-split optimizer")
            optimizer, muon_transition = build_torch_muon_adamw(
                model,
                flags,
                total_epochs=args.epochs,
                start_epoch=epoch,
                outgoing_adamw=optimizer,
            )
            _emit_trajectory_row(
                out,
                {
                    "stage": "muon_transition",
                    "epoch": epoch,
                    **muon_transition,
                    **optimizer.set_epoch(epoch),
                    "state_policy": "warm-start Muon first moment; reset remaining state",
                    "backend": "torch_cuda",
                    "authority": "[contest-CUDA training-advisory]",
                    "promotion_eligible": False,
                },
            )
        elif controller_step.muon_on and not isinstance(optimizer, TorchMuonAdamW):
            raise RuntimeError(
                "resumed/fired Muon controller lacks the matching split optimizer state"
            )
        if isinstance(optimizer, TorchMuonAdamW):
            optimizer.set_epoch(epoch)
        tail_step = None
        tail_stop_after_epoch = False
        if (
            tail_controller is not None
            and controller_step.muon_on
            and epoch >= tail_start_epoch
        ):
            tail_step = tail_controller.step(
                epoch,
                list(controller.dseg_history),
                live_mq=None,
                byte_rows=None,
            )
            with torch.no_grad():
                model.softmax_temp.fill_(float(tail_step.tau))
            for group in optimizer.param_groups:
                group["lr"] = float(tail_step.lr)
            if tail_step.begin_cycle:
                tail_blob = _checkpoint_blob(
                    model,
                    ema,
                    optimizer,
                    epoch - 1,
                    cfg_hash,
                    dsl_argv,
                    pair_cursor=pair_cursor,
                    controller_state=controller.state_dict(),
                    protected_seed=protected_seed,
                    seed_optimizer=seed_optimizer,
                    tail_controller=tail_controller,
                    scorer_sha256=scorer_sha256,
                )
                _atomic_torch_save(
                    tail_blob,
                    out / "stage_checkpoints"
                    / f"ep{epoch - 1:05d}_{tail_step.stage_tag}_muon.pt",
                )
                _emit_trajectory_row(
                    out,
                    {
                        "stage": "tail_cycle_begin",
                        "epoch": epoch,
                        "cycle": tail_step.cycle_k,
                        "tau": tail_step.tau,
                        "lr": tail_step.lr,
                        "reason": tail_step.reason,
                        "rate_aware": False,
                        "backend": "torch_cuda",
                        "authority": "[contest-CUDA training-advisory]",
                        "promotion_eligible": False,
                    },
                )
            if tail_step.stop:
                tail_stop_after_epoch = True
                _emit_trajectory_row(
                    out,
                    {
                        "stage": "tail_powerplay_stop",
                        "epoch": epoch,
                        "cycle": tail_step.cycle_k,
                        "reason": tail_step.reason,
                        "net_marginal_s_per_ep": (
                            None if tail_step.marginal != tail_step.marginal
                            else float(tail_step.marginal)
                        ),
                        "rate_aware": False,
                        "backend": "torch_cuda",
                        "authority": "[contest-CUDA training-advisory]",
                        "promotion_eligible": False,
                    },
                )
        if bool(flags.get("--ladder-island-homotopy", False)):
            from tac.witness_curriculum.ladder_homotopy import ARM_LANE, ARM_MOVABLE

            island_targets.refresh_amplify_(
                lane_px=int(controller_step.ladder_rungs[ARM_LANE]),
                movable_px=int(controller_step.ladder_rungs[ARM_MOVABLE]),
            )
        birth_active = bool(controller.birth.fired)
        effective_logit_offsets = birth_scaled_logit_offsets(
            base_logit_offsets, controller_step.birth_multipliers
        )
        seed_weight = seed_compose_weight_at_epoch(
            int(flags["--seed-anneal-epochs"]),
            str(flags["--seed-anneal-shape"]),
            epoch,
        )
        for controller_row in controller_step.telemetry:
            # Birth rows are emitted with the completed epoch below. The runtime
            # also carries them into begin_epoch so a crash cannot lose them.
            if controller_row.get("stage") == "birth_completion":
                continue
            _emit_trajectory_row(
                out,
                {
                    **controller_row,
                    "backend": "torch_cuda",
                    "authority": "[contest-CUDA training-advisory]",
                    "promotion_eligible": False,
                },
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        epoch_train_t0 = time.perf_counter()
        pair_cursor.begin_epoch(epoch)
        ep_acc = ep_tot = 0
        chunk_index = 0
        while not pair_cursor.epoch_complete():
            chunk = pair_cursor.next_epoch_indices(int(flags["--accum-pairs"]))
            chunk_np = np.asarray(chunk, dtype=np.int64)
            chunk_index_tensor = torch.as_tensor(chunk, device=device, dtype=torch.long)
            if compile_receipts and hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                torch.compiler.cudagraph_mark_step_begin()
            # MAX-throughput steady state: exactly one batched witness/carrier
            # dispatch and one batch through each frozen scorer per accum chunk.
            dispatch = _generated_pose_pair_dispatch(
                model,
                feats,
                chunk,
                pose_carrier,
                cfg,
                lane_band=(lane_band if controller_step.lane_band_on else None),
                protected_seed=protected_seed,
                seed_weight=seed_weight,
                return_witness_alone=witness_alone_island,
                r_operator=compiled_r_operator,
            )
            if witness_alone_island:
                frames, phi, witness_alone_frames = dispatch
                scorer_frames = torch.cat((frames, witness_alone_frames), dim=0)
            else:
                frames, phi = dispatch
                witness_alone_frames = None
                scorer_frames = frames
            seg_in = seg.preprocess_input(
                scorer_frames.permute(0, 1, 4, 2, 3).contiguous()
            )
            scorer_logits = seg(seg_in)
            if witness_alone_island:
                logits, formation_logits = scorer_logits.split(len(chunk), dim=0)
            else:
                logits = formation_logits = scorer_logits
            target = lstars_device.index_select(0, chunk_index_tensor)
            offsets = effective_logit_offsets.to(dtype=logits.dtype)
            adjusted_logits = logits + offsets[None, :, None, None]
            tau = max(0.31, 0.31 ** (epoch / max(args.epochs, 1)))
            seg_per_pixel = (
                tau * torch.logsumexp(adjusted_logits / tau, dim=1)
                - adjusted_logits.gather(1, target[:, None]).squeeze(1)
            )
            gt_margin = margins_device.index_select(0, chunk_index_tensor).to(seg_per_pixel.dtype)
            controller.observe_scorer_chunk(
                formation_logits.argmax(dim=1), target, gt_margin
            )
            seg_weight = 1.0 + 4.0 * torch.exp(-gt_margin.clamp_min(0.0))
            seg_loss = (seg_per_pixel * seg_weight).mean()
            pose6 = _pose6(pose, frames)
            pose_target = gt_poses_device.index_select(0, chunk_index_tensor).to(pose6.dtype)
            pose_loss = loss_ops["pose"](pose6, pose_target)
            eik, length = loss_ops["eikonal_length"](phi)
            seg_contrib = float(flags["--w-seg"]) * seg_loss
            pose_weight = (
                float(flags["--w-pose"]) if controller_step.pose_finish_on else 0.0
            )
            pose_contrib = pose_weight * pose_loss
            eik_contrib = float(flags["--eikonal-weight"]) * eik
            length_contrib = float(flags["--length-weight"]) * length
            raw_logits_nhwc = formation_logits.permute(0, 2, 3, 1).contiguous()
            signed = loss_ops["realized_margin"](raw_logits_nhwc, target)
            island_weight = island_targets.weight.index_select(0, chunk_index_tensor).to(signed.dtype)
            if birth_active:
                lane_mask = island_targets.lane_mask.index_select(0, chunk_index_tensor)
                movable_mask = island_targets.movable_mask.index_select(0, chunk_index_tensor)
                amplify_loss = loss_ops["island_birth_perclass"](
                    signed,
                    island_weight,
                    lane_mask,
                    movable_mask,
                    float(flags.get("--amplify-margin-target", 1.0)),
                    float(controller_step.birth_multipliers.get(island_detection.lane_cls, 1.0)),
                    float(controller_step.birth_multipliers.get(island_detection.movable_cls, 1.0)),
                    form=str(flags.get("--amplify-form", "hinge")),
                )
            else:
                amplify_loss = loss_ops["island_birth"](
                    signed,
                    island_weight,
                    float(flags.get("--amplify-margin-target", 1.0)),
                    form=str(flags.get("--amplify-form", "hinge")),
                )
            amplify = float(flags.get("--amplify-weight", 0.0)) * amplify_loss
            persist_scale = min(
                1.0, epoch / max(1, int(flags.get("--persistence-warmup-epochs", 0)))
            )
            persistence = (
                float(flags.get("--persistence-loss-weight", 0.0)) * persist_scale
                * loss_ops["persistence"](
                    raw_logits_nhwc,
                    target,
                    persist_classes,
                    recall_weight=float(flags.get("--persistence-recall-weight", 1.0)),
                    recall_class_scale=(
                        tuple(
                            float(controller_step.birth_multipliers.get(cls, 1.0))
                            for cls in persist_classes
                        )
                        if birth_active
                        else None
                    ),
                )
            )
            area = loss_ops["area"](raw_logits_nhwc, target, area_lambdas)
            _entropy_bits, entropy_rate = loss_ops["weight_entropy"](
                model, sigma=float(flags.get("--weight-entropy-penalty-sigma", 0.2))
            )
            weight_entropy = (
                float(flags.get("--weight-entropy-penalty-lambda", 0.0)) * entropy_rate
            )
            phase_advect = torch.zeros((), device=device)
            if phase_weight > 0.0 and epoch >= phase_start:
                phase_rows = [phase_provider(pi) for pi in chunk]
                ref = torch.as_tensor(
                    np.stack([r[0] for r in phase_rows]), device=device, dtype=signed.dtype
                )
                direction = torch.as_tensor(
                    np.stack([r[1] for r in phase_rows]), device=device, dtype=signed.dtype
                )
                pw = torch.as_tensor(
                    np.stack([r[2] for r in phase_rows]), device=device, dtype=signed.dtype
                )
                tie = witness_tie_coordinate_torch(signed, direction)
                phase_num = ((tie - ref).square() * pw).sum(dim=(-2, -1))
                phase_den = pw.sum(dim=(-2, -1)) + 1e-6
                phase_advect = phase_weight * (phase_num / phase_den).mean()
            temporal_screw = torch.zeros((), device=device)
            if temporal_weight > 0.0 and epoch >= temporal_start:
                frame0 = frames[:, 0:1]
                seg0_in = seg.preprocess_input(frame0.permute(0, 1, 4, 2, 3).contiguous())
                logits0 = seg(seg0_in).permute(0, 2, 3, 1).contiguous()
                prob0 = torch.softmax(logits0, dim=-1)[..., list(temporal_classes)]
                prob1 = torch.softmax(raw_logits_nhwc, dim=-1)[..., list(temporal_classes)]
                for pi in chunk:
                    if pi not in temporal_grid_cache:
                        assert temporal_geom is not None
                        temporal_grid_cache[pi] = homography_grid_from_xi(
                            temporal_xi[pi], temporal_geom, device=device, dtype=prob0.dtype
                        )
                grid = torch.cat([temporal_grid_cache[pi][0] for pi in chunk], dim=0)
                valid = torch.cat([temporal_grid_cache[pi][1] for pi in chunk], dim=0)
                warped0 = warp_field_persist_torch(prob0, grid, valid)
                annulus = torch.as_tensor(
                    margins[chunk_np] < float(flags.get("--seg-temporal-screw-band", 2.0)),
                    device=device, dtype=prob0.dtype,
                )
                temporal_num = (
                    (prob1 - warped0).square().sum(-1) * annulus
                ).sum(dim=(-2, -1))
                temporal_den = annulus.sum(dim=(-2, -1)) + 1e-6
                temporal_screw = temporal_weight * (temporal_num / temporal_den).mean()
            chroma = torch.zeros((), device=device)
            if controller_step.chroma_on:
                assert gt_f1_render_device is not None
                gt = gt_f1_render_device.index_select(0, chunk_index_tensor).to(frames.dtype)
                ann = margins_device.index_select(0, chunk_index_tensor) < float(
                    flags["--seg-chroma-boundary-margin-band"]
                )
                chroma = loss_ops["chroma"](frames[:, 1], gt, ann)
            chroma_contrib = float(flags["--seg-chroma-boundary-weight"]) * chroma
            total = (
                seg_contrib + pose_contrib + eik_contrib + length_contrib
                + amplify + persistence + area + weight_entropy + phase_advect
                + temporal_screw + chroma_contrib
            )
            optimizer.zero_grad(set_to_none=True)
            if seed_optimizer is not None:
                seed_optimizer.zero_grad(set_to_none=True)
            ep_tot += 1
            weights_stepped = bool(torch.isfinite(total.detach()))
            group_norms = {}
            if weights_stepped:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(total).backward()
                    scaler.unscale_(optimizer)
                    if seed_optimizer is not None:
                        scaler.unscale_(seed_optimizer)
                else:
                    total.backward()
                if protected_seed is not None:
                    protected_seed.contain_grad_()
                    if (
                        protected_seed.residual.grad is not None
                        and not bool(torch.isfinite(protected_seed.residual.grad).all())
                    ):
                        weights_stepped = False
                try:
                    group_norms = clip_grad_groups(
                        parameter_groups(model), float(flags["--grad-clip"])
                    )
                except RuntimeError:
                    weights_stepped = False
            if weights_stepped:
                if scaler is not None and scaler.is_enabled():
                    scaler.step(optimizer)
                    if seed_optimizer is not None:
                        scaler.step(seed_optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                    if seed_optimizer is not None:
                        seed_optimizer.step()
                _ema_update(ema, model, ema_decay)
                ep_acc += 1
                pair_cursor.record_accepted(1)
            else:
                optimizer.zero_grad(set_to_none=True)
                if seed_optimizer is not None:
                    seed_optimizer.zero_grad(set_to_none=True)
            # One telemetry synchronization per exhaustive epoch, never per
            # accum-pair chunk. Controller counters remain on device too.
            if pair_cursor.epoch_complete():
                finite_norms = [v for v in group_norms.values() if v is not None]
                gnorm = max((float(v) for v in finite_norms), default=0.0)
                terms = {
                    "seg": float(seg_contrib.detach()),
                    "pose": float(pose_contrib.detach()),
                    "eikonal": float(eik_contrib.detach()),
                    "length": float(length_contrib.detach()),
                    "chroma_boundary": float(chroma_contrib.detach()),
                    "island_amplify": float(amplify.detach()),
                    "area_constraint": float(area.detach()),
                    "persistence": float(persistence.detach()),
                    "weight_entropy": float(weight_entropy.detach()),
                    "phase_advect": float(phase_advect.detach()),
                    "temporal_screw": float(temporal_screw.detach()),
                }
                telemetry = loss_terms_row(
                    epoch=epoch, accum_batch=chunk_index, terms=terms,
                    total=float(total.detach()), gnorm=round(gnorm, 4),
                    accepted_frac=ep_acc / ep_tot, weights_stepped=weights_stepped,
                    hosc_beta=round(float(model.hosc_beta), 4),
                    softmax_temp=round(float(model.softmax_temp), 4), backend="torch_cuda",
                    curriculum_stage=stage, lr=optimizer.param_groups[0]["lr"],
                    pairs=chunk, pair_count=len(chunk), vectorized_chunk=True,
                    telemetry_scope="epoch_final_chunk",
                    authority="[contest-CUDA training-advisory]", promotion_eligible=False,
                    wall_clock_s=time.time() - t0,
                )
                telemetry["controller"] = {
                    "muon_on": controller_step.muon_on,
                    "lane_band_on": controller_step.lane_band_on,
                    "chroma_on": controller_step.chroma_on,
                    "pose_finish_on": controller_step.pose_finish_on,
                    "pose_banked_r1": controller_step.pose_banked_r1,
                    "effective_pose_weight": pose_weight,
                }
                _stage_epoch_row(pending_epoch_rows, telemetry)
            chunk_index += 1

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        epoch_train_seconds = time.perf_counter() - epoch_train_t0
        _stage_epoch_row(
            pending_epoch_rows,
            {
                "stage": "training_throughput_epoch",
                "epoch": epoch,
                "pairs": args.num_pairs,
                "optimizer_updates": ep_acc,
                "optimizer_updates_attempted": ep_tot,
                "optimizer_updates_successful": ep_acc,
                "seconds": round(epoch_train_seconds, 6),
                "pairs_per_second": round(args.num_pairs / max(epoch_train_seconds, 1e-12), 6),
                "updates_per_second": round(ep_acc / max(epoch_train_seconds, 1e-12), 6),
                "productive_updates_per_second": round(
                    ep_acc / max(epoch_train_seconds, 1e-12), 6
                ),
                "peak_allocated_bytes": (
                    int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
                ),
                "amp_dtype": execution_policy.amp_dtype,
                "compiled_regions": bool(compile_receipts),
                "cuda_graph_trees_requested": bool(
                    compile_receipts and execution_policy.cuda_graphs
                ),
                "compile_warmup_epoch": epoch == start + 1,
                "authority": "[contest-CUDA training-throughput]"
                if device.type == "cuda" else "[macOS-CPU/Torch advisory]",
                "promotion_eligible": False,
                "score_claim": False,
            }
        )

        controller_epoch = controller.end_epoch(epoch)
        for birth_row in controller_epoch["birth_telemetry"]:
            _stage_epoch_row(
            pending_epoch_rows,
                {
                    **birth_row,
                    "backend": "torch_cuda",
                    "authority": "[contest-CUDA training-advisory]",
                    "promotion_eligible": False,
                }
            )

        probe_indices = _jacobian_probe_pair_indices(flags, gt_poses, epoch)
        if probe_indices:
            with torch.no_grad():
                _probe_frames, _probe_phi, probe_inputs = _generated_pose_pair_dispatch(
                    model,
                    feats,
                    probe_indices,
                    pose_carrier,
                    cfg,
                    lane_band=(lane_band if controller_step.lane_band_on else None),
                    protected_seed=protected_seed,
                    seed_weight=seed_weight,
                    return_probe_inputs=True,
                    r_operator=compiled_r_operator,
                )

            def pose_from_native_frames(native_frame0, scored_frame1):
                scored_frame0 = torch.nn.functional.interpolate(
                    native_frame0.permute(0, 3, 1, 2),
                    size=(cfg.render_h, cfg.render_w),
                    mode="bilinear",
                    align_corners=False,
                ).permute(0, 2, 3, 1).contiguous()
                return _pose6(
                    pose, torch.stack((scored_frame0, scored_frame1), dim=1)
                )

            conditioning = torch_pose_jacobian_conditioning(
                pose_from_native_frames,
                pose_carrier,
                probe_inputs["native0"],
                probe_inputs["scored1"],
                probe_indices,
                sigma_floor=float(flags["--jacobian-basin-sigma-floor"]),
            )
            pose_verdict = controller.observe_sigma_min(
                epoch, float(conditioning["median_sigma_min"])
            )
            _stage_epoch_row(
            pending_epoch_rows,
                {
                    "stage": "jacobian_basin_t1",
                    "epoch": epoch,
                    **conditioning,
                    "pose_gate": pose_verdict,
                    "cadence_epochs": max(1, int(flags["--eval-every"]))
                    * max(1, int(flags["--jacobian-basin-every"])),
                    "backend": "torch_cuda",
                    "authority": "[contest-CUDA training-advisory]",
                    "promotion_eligible": False,
                }
            )

        polyak_observed = controller.observe_polyak(epoch, model)
        if polyak_observed:
            _stage_epoch_row(
            pending_epoch_rows,
                {
                    "stage": "polyak_finisher_observe",
                    "epoch": epoch,
                    "count": int(controller.polyak.count),
                    "start_epoch": int(controller.polyak.start_epoch),
                    "backend": "torch_cuda",
                    "authority": "[contest-CUDA training-advisory]",
                    "promotion_eligible": False,
                }
            )
        _stage_epoch_row(
            pending_epoch_rows,
            {
                "stage": "v9_controller_epoch",
                **{key: value for key, value in controller_epoch.items()
                   if key != "birth_telemetry"},
                "muon_on": controller.muon_on,
                "pose_banked_r1": controller.pose_banked_r1,
                "backend": "torch_cuda",
                "authority": "[contest-CUDA training-advisory]",
                "promotion_eligible": False,
            }
        )

        blob = None
        last_completed_epoch = epoch
        if _canonical_checkpoint_due(
            epoch,
            ckpt_every=ckpt_every,
            run_end_epoch=run_end_epoch,
            runtime_stop_after_epochs=args.stop_after_epochs,
            tail_stop_after_epoch=tail_stop_after_epoch,
        ):
            controller_state = controller.state_dict()
            blob = _checkpoint_blob(
                model, ema, optimizer, epoch, cfg_hash, dsl_argv,
                pair_cursor=pair_cursor, controller_state=controller_state,
                protected_seed=protected_seed, seed_optimizer=seed_optimizer,
                tail_controller=tail_controller, scorer_sha256=scorer_sha256,
            )
            _atomic_torch_save(blob, out / TORCH_RESUME_PT)
            _atomic_torch_save({"schema": "v9_cgauge_torch_ema_v1", "epoch": epoch,
                                "ema": ema.state_dict(), "config_hash": cfg_hash,
                                "scorer_sha256": scorer_sha256,
                                "dsl_argv": list(dsl_argv)}, out / TORCH_EMA_PT)
            latest_polyak = _polyak_checkpoint_blob(
                controller, epoch, cfg_hash, dsl_argv, scorer_sha256
            )
            if latest_polyak is not None:
                _atomic_torch_save(latest_polyak, out / _TORCH_POLYAK_PT)
            _flush_trajectory_rows(out, pending_epoch_rows)
        if tail_stop_after_epoch:
            tail_terminated = True
            break
    controller_state = controller.state_dict()
    final_blob = _checkpoint_blob(
        model, ema, optimizer, last_completed_epoch, cfg_hash, dsl_argv,
        pair_cursor=pair_cursor, controller_state=controller_state,
        protected_seed=protected_seed, seed_optimizer=seed_optimizer,
        tail_controller=tail_controller, scorer_sha256=scorer_sha256,
    )
    _atomic_torch_save(
        final_blob,
        out / "stage_checkpoints" / f"ep{last_completed_epoch:05d}_{prev_stage}.pt",
    )
    final_polyak = _polyak_checkpoint_blob(
        controller, last_completed_epoch, cfg_hash, dsl_argv, scorer_sha256
    )
    if final_polyak is not None:
        _atomic_torch_save(final_polyak, out / _TORCH_POLYAK_PT)
        _atomic_torch_save(
            final_polyak,
            out / "stage_checkpoints"
            / f"ep{last_completed_epoch:05d}_{prev_stage}_polyak.pt",
        )
    _atomic_json(
        {
            "schema": "v9_cgauge_torch_train_result_v1",
            "status": (
                "completed"
                if last_completed_epoch >= args.epochs
                else "governed_tail_stop"
                if tail_terminated
                else "runtime_epoch_budget_reached"
            ),
            "backend": "torch_cuda",
            "epochs_completed": last_completed_epoch,
            "runtime_epochs_completed": last_completed_epoch - start,
            "runtime_stop_after_epochs": args.stop_after_epochs,
            "typed_total_epochs": args.epochs,
            "dsl_config": args.dsl_config,
            "config_hash": cfg_hash,
            "scorer_sha256": scorer_sha256,
            "seed": int(flags["--seed"]),
            "polyak_candidate": (
                _TORCH_POLYAK_PT if final_polyak is not None else None
            ),
            "authority": "[contest-CUDA training-advisory] NON-PROMOTABLE",
            "pointer_delta": "none",
            "completed_at_utc": _utc(),
        },
        out / TORCH_TRAIN_RESULT_JSON,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
