"""Multi-model inflation: dispatch different postfilters to different frames (Eureka 4).

The key insight: a single postfilter cannot optimally correct ALL frames.
The "easy" frames (smooth highway, good lighting) need gentle correction,
while "hard" frames (lane changes, shadows, occlusions) need aggressive
correction that would harm easy frames if applied globally.

Solution: train two models:
  - Model A: conservative, trained on all frames, deployed globally.
  - Model B: aggressive, trained on the hardest ~50 frames, applied only there.

At inflate time, Model A runs on all frames, then Model B overwrites its
output on the hard frames. Both models are tiny (<50KB each), so total
archive cost is ~100KB.

Usage::

    from tac.multi_model_inflate import MultiModelInflater, build_multi_model_archive

    # Build archive with both models
    build_multi_model_archive("model_a.pt", "model_b.pt", hard_frames=[12, 45, 89, ...])

    # At inflate time
    inflater = MultiModelInflater.from_archive(archive_dir)
    corrected = inflater.inflate(frames)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


def _log(msg: str, verbose: bool = True) -> None:
    """Print to stderr if verbose."""
    if verbose:
        print(f"  [multi_model] {msg}", file=sys.stderr, flush=True)


class MultiModelInflater:
    """Dispatch postfilter inference across multiple models.

    Model A runs on ALL frames. Model B runs ONLY on specified hard frames
    and overwrites Model A's output for those frames. This gives hard frames
    extra correction without penalizing easy frames.

    Supports N models: one global (Model A) and any number of frame-specific
    override models. Override models are applied in priority order (last
    model in the list wins for overlapping frames).

    Attributes:
        model_a: primary postfilter (runs on all frames).
        override_models: list of (model, frame_set) pairs applied in order.
        device: computation device.
    """

    def __init__(
        self,
        models: list[tuple[nn.Module, list[int] | None]],
        device: str | torch.device = "cpu",
        verbose: bool = True,
    ) -> None:
        """Initialize with a list of (model, frame_indices) pairs.

        Args:
            models: list of (model, frame_indices) tuples. frame_indices=None
                means "all frames". The first model with frame_indices=None is
                the global model. Models with specific indices override the
                global model for those frames. Multiple override models are
                supported — later entries take priority for overlapping frames.
            device: computation device.
            verbose: print progress.
        """
        self.device = torch.device(device) if isinstance(device, str) else device
        self.verbose = verbose
        self.model_a: nn.Module | None = None
        self.override_models: list[tuple[nn.Module, set[int]]] = []

        for model, indices in models:
            model = model.to(self.device).eval()
            if indices is None:
                self.model_a = model
            else:
                self.override_models.append((model, set(indices)))

        if self.model_a is None and len(models) > 0:
            # First model is implicitly global if no None-index model specified
            self.model_a = models[0][0].to(self.device).eval()

        # Backward compat: expose model_b and hard_frames for 2-model case
        self.model_b = self.override_models[0][0] if self.override_models else None
        self.hard_frames = self.override_models[0][1] if self.override_models else set()

    @classmethod
    def from_archive(
        cls,
        archive_dir: str | Path,
        device: str | torch.device = "cpu",
        verbose: bool = True,
    ) -> "MultiModelInflater":
        """Load multi-model inflater from an archive directory.

        Looks for:
          - postfilter_int8.pt (Model A, required)
          - postfilter_hard.pt (Model B, optional)
          - hard_frames.json (frame indices for Model B, optional)

        Args:
            archive_dir: path to extracted archive directory.
            device: computation device.
            verbose: print progress.

        Returns:
            MultiModelInflater instance.
        """
        from tac.quantization import load_postfilter_int8

        archive_dir = Path(archive_dir)
        models: list[tuple[nn.Module, list[int] | None]] = []

        # Model A: required
        model_a_path = archive_dir / "postfilter_int8.pt"
        if not model_a_path.exists():
            raise FileNotFoundError(f"Model A not found: {model_a_path}")

        model_a = load_postfilter_int8(str(model_a_path), device=device)
        models.append((model_a, None))

        if verbose:
            _log(f"Loaded Model A from {model_a_path}")

        # Model B: optional
        model_b_path = archive_dir / "postfilter_hard.pt"
        hard_frames_path = archive_dir / "hard_frames.json"

        if model_b_path.exists() and hard_frames_path.exists():
            model_b = load_postfilter_int8(str(model_b_path), device=device)
            with open(hard_frames_path) as f:
                hard_frames = json.load(f)

            if not isinstance(hard_frames, list):
                _log(f"WARNING: hard_frames.json is not a list, ignoring Model B", verbose)
            else:
                models.append((model_b, hard_frames))
                if verbose:
                    _log(f"Loaded Model B from {model_b_path}: {len(hard_frames)} hard frames")
        elif model_b_path.exists():
            _log("Model B found but no hard_frames.json, ignoring", verbose)
        elif verbose:
            _log("No Model B found, using single-model mode")

        return cls(models, device=device, verbose=verbose)

    def inflate_batch(
        self,
        frames_bchw: torch.Tensor,
        frame_offset: int = 0,
    ) -> torch.Tensor:
        """Run multi-model inference on a batch of frames.

        Model A processes all frames. Then Model B overwrites its output
        for any frames in the hard_frames set.

        Args:
            frames_bchw: (B, 3, H, W) float tensor in [0, 255].
            frame_offset: global frame index of the first frame in this batch.

        Returns:
            (B, 3, H, W) corrected frames.
        """
        B = frames_bchw.shape[0]
        x = frames_bchw.to(self.device)

        # Model A: all frames
        with torch.inference_mode():
            if self.model_a is not None:
                out = self.model_a(x)
            else:
                out = x.clone()

        # Override models: apply in order (later entries win for overlapping frames)
        for model_idx, (override_model, frame_set) in enumerate(self.override_models):
            override_local = []
            for i in range(B):
                global_idx = frame_offset + i
                if global_idx in frame_set:
                    override_local.append(i)

            if override_local:
                override_indices = torch.tensor(override_local, dtype=torch.long)
                override_input = x[override_indices]
                with torch.inference_mode():
                    override_output = override_model(override_input)
                out[override_indices] = override_output

                if self.verbose:
                    _log(f"Override model {model_idx} applied to {len(override_local)} frames")

        return out

    def inflate(
        self,
        frames_bchw: torch.Tensor,
        batch_size: int = 8,
    ) -> torch.Tensor:
        """Run multi-model inference on all frames with batching.

        Args:
            frames_bchw: (N, 3, H, W) float tensor in [0, 255].
            batch_size: frames per forward pass.

        Returns:
            (N, 3, H, W) corrected frames.
        """
        N = frames_bchw.shape[0]
        results = []

        for batch_start in range(0, N, batch_size):
            batch_end = min(batch_start + batch_size, N)
            batch = frames_bchw[batch_start:batch_end]
            out = self.inflate_batch(batch, frame_offset=batch_start)
            results.append(out.cpu())

        return torch.cat(results, dim=0)


def build_multi_model_archive(
    model_a_path: str | Path,
    model_b_path: str | Path,
    hard_frames: list[int],
    output_dir: str | Path | None = None,
    verbose: bool = True,
) -> Path:
    """Package both models and hard-frame list into an archive directory.

    Copies Model A as postfilter_int8.pt, Model B as postfilter_hard.pt,
    and writes hard_frames.json alongside them.

    Args:
        model_a_path: path to Model A (global postfilter) int8 checkpoint.
        model_b_path: path to Model B (hard-frame postfilter) int8 checkpoint.
        hard_frames: list of frame indices where Model B should be applied.
        output_dir: directory to write files to. If None, uses model_a_path's
            parent directory.
        verbose: print progress.

    Returns:
        Path to the output directory.
    """
    import shutil

    model_a_path = Path(model_a_path)
    model_b_path = Path(model_b_path)

    if output_dir is None:
        output_dir = model_a_path.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy models
    dest_a = output_dir / "postfilter_int8.pt"
    dest_b = output_dir / "postfilter_hard.pt"
    dest_frames = output_dir / "hard_frames.json"

    if model_a_path != dest_a:
        shutil.copy2(str(model_a_path), str(dest_a))
    shutil.copy2(str(model_b_path), str(dest_b))

    with open(dest_frames, "w") as f:
        json.dump(sorted(hard_frames), f)

    if verbose:
        a_size = dest_a.stat().st_size
        b_size = dest_b.stat().st_size
        _log(f"Archive: Model A {a_size} bytes, Model B {b_size} bytes, "
             f"{len(hard_frames)} hard frames")

    return output_dir


def estimate_multi_model_savings(
    n_frames: int,
    n_hard: int,
    model_a_time_per_frame: float = 0.005,
    model_b_time_per_frame: float = 0.008,
) -> dict[str, float]:
    """Estimate time and quality savings from multi-model dispatch.

    Args:
        n_frames: total number of frames.
        n_hard: number of hard frames getting Model B.
        model_a_time_per_frame: seconds per frame for Model A.
        model_b_time_per_frame: seconds per frame for Model B.

    Returns:
        dict with estimated times and savings.
    """
    single_model_time = n_frames * model_a_time_per_frame
    multi_model_time = (
        n_frames * model_a_time_per_frame  # Model A on all
        + n_hard * model_b_time_per_frame   # Model B on hard frames
    )
    overhead = multi_model_time - single_model_time

    return {
        "single_model_seconds": single_model_time,
        "multi_model_seconds": multi_model_time,
        "overhead_seconds": overhead,
        "overhead_percent": 100.0 * overhead / single_model_time if single_model_time > 0 else 0.0,
        "hard_frame_coverage": 100.0 * n_hard / n_frames if n_frames > 0 else 0.0,
    }
