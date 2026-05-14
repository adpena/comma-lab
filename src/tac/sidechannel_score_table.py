# SPDX-License-Identifier: MIT
"""Shared helpers for PR106 sidechannel score-table producers.

These utilities keep compress-time scorer-table scripts aligned on custody,
checkpoint, and objective semantics. They are intentionally scorer-free at
inflate time; scorers are loaded only by producer scripts before bytes are
emitted into an archive.
"""
from __future__ import annotations

import gc
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tac.deploy.claims import active_claim_row


def repo_root_from_tac() -> Path:
    return Path(__file__).resolve().parents[2]


def atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def atomic_save_npy(path: Path, array: np.ndarray) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        np.save(f, array, allow_pickle=False)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def read_json_object(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def completed_prefix_rows(table: np.ndarray) -> int:
    """Return the finite scored-row prefix, rejecting non-prefix checkpoints."""
    if table.ndim != 2:
        raise ValueError(f"score table checkpoint must be 2-D, got shape {table.shape}")
    finite_rows = np.isfinite(table).all(axis=1)
    incomplete = np.flatnonzero(~finite_rows)
    if incomplete.size == 0:
        return int(table.shape[0])
    first_incomplete = int(incomplete[0])
    if finite_rows[first_incomplete:].any():
        raise ValueError("score table checkpoint has non-prefix finite rows")
    return first_incomplete


def resume_safe_prefix_pairs(table: np.ndarray) -> int:
    """Return a pair-aligned row prefix for frame-indexed score tables."""
    complete = completed_prefix_rows(table)
    if complete < int(table.shape[0]) and complete % 2:
        return complete - 1
    return complete


def verify_active_lane_claim(
    claims_path: Path,
    *,
    lane_id: str,
    instance_job_id: str,
) -> dict[str, str]:
    """Return the newest matching active claim row or raise ValueError."""
    return active_claim_row(
        claims_path,
        lane_id=lane_id,
        instance_job_id=instance_job_id,
    )


def score_without_rate(pose_dist: torch.Tensor, seg_dist: torch.Tensor) -> torch.Tensor:
    """Contest objective without the archive-rate constant."""
    return 100.0 * seg_dist + torch.sqrt(torch.clamp(10.0 * pose_dist, min=0.0))


def is_cuda_oom(exc: BaseException) -> bool:
    """Return true for PyTorch CUDA OOM failures without catching CPU/parser OOMs."""
    oom_type = getattr(torch, "OutOfMemoryError", None)
    if oom_type is not None and isinstance(exc, oom_type):
        return True
    text = f"{type(exc).__name__}: {exc}".lower()
    return "cuda" in text and ("out of memory" in text or "outofmemoryerror" in text)


def clear_cuda_retry_state(device: torch.device) -> None:
    """Release Python and CUDA allocator state before retrying a smaller tile."""
    gc.collect()
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_distortion_net(device: torch.device, *, repo_root: Path | None = None):
    root = repo_root if repo_root is not None else repo_root_from_tac()
    upstream_dir = root / "upstream"
    sys.path.insert(0, str(upstream_dir))
    from modules import DistortionNet, posenet_sd_path, segnet_sd_path  # type: ignore[import-not-found]

    net = DistortionNet().eval().to(device=device)
    net.load_state_dicts(posenet_sd_path, segnet_sd_path, device)
    return net


def load_gt_dataloader(
    *,
    device: torch.device,
    video_names_file: Path,
    uncompressed_dir: Path,
    batch_pairs: int,
    seed: int,
    num_threads: int,
    prefetch_queue_depth: int,
    repo_root: Path | None = None,
):
    root = repo_root if repo_root is not None else repo_root_from_tac()
    upstream_dir = root / "upstream"
    sys.path.insert(0, str(upstream_dir))
    from frame_utils import DaliVideoDataset  # type: ignore[import-not-found]

    names = [line.strip() for line in video_names_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    ds_gt = DaliVideoDataset(
        names,
        data_dir=uncompressed_dir,
        batch_size=batch_pairs,
        device=device,
        num_threads=num_threads,
        seed=seed,
        prefetch_queue_depth=prefetch_queue_depth,
    )
    ds_gt.prepare_data()
    return torch.utils.data.DataLoader(ds_gt, batch_size=None, num_workers=0)
