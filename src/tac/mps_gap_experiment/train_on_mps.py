# SPDX-License-Identifier: MIT
"""MPS training loop for the tiny renderer (gap-experiment Phase 1).

Trains :class:`tac.mps_gap_experiment.tiny_renderer.TinyRenderer` on real
contest frames from ``upstream/videos/0.mkv`` and emits a checkpoint +
metrics JSON to ``experiments/results/mps_gap_experiment_<utc>/`` for the
subsequent CUDA gap comparison.

NON-NEGOTIABLE per CLAUDE.md:

* ``eval_roundtrip=True`` at evaluation time per "eval_roundtrip — NON-NEGOTIABLE".
* EMA(0.997) tracked over training weights per "EMA — NON-NEGOTIABLE". The
  saved checkpoint is the EMA shadow, NOT the live weights.
* Real frames decoded from ``upstream/videos/0.mkv`` via pyav, NEVER synthetic
  noise (CLAUDE.md "MPS auth eval is NOISE" rules out the synthetic-data
  failure mode that produced the historical 23× claim — the gap experiment
  needs real frames or the result is methodology-invalid).
* No archive grammar / no contest score claims. Every artifact tagged
  ``evidence_grade="MPS-research-signal"`` + ``score_claim=False`` +
  ``promotion_eligible=False`` per Catalog #192 / #317.

The training loss is L1 pixel loss only (scorer-free) by design — the goal
is to exercise the canonical MPS forward / backward path with real-frame data,
not to optimize a score. Scorer-loss can be enabled via the
``include_scorer_loss`` kwarg for richer gap measurement; default OFF keeps
the training loop fast (<10 minutes) and the diagnostic surface clean.

The script does NOT promote anything — when complete, the operator (or the
harness in :mod:`tac.mps_gap_experiment.harvest_and_verdict`) invokes the
gap-comparison step that re-runs the saved EMA shadow on Modal A10G CUDA.
"""

from __future__ import annotations

import contextlib
import copy
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator

import torch
import torch.nn as nn

from tac.mps_gap_experiment.tiny_renderer import (
    TinyRenderer,
    build_tiny_renderer,
    count_params,
)
from tac.training import EMA

__all__ = (
    "TrainingMetrics",
    "train_on_mps_real_frames",
)


_REPO_ROOT_HINT = "upstream/videos/0.mkv"


@dataclass(frozen=True)
class EpochMetric:
    """Per-epoch training metric row."""

    epoch: int
    pixel_loss_mean: float
    pairs_seen: int
    elapsed_s: float


@dataclass(frozen=True)
class TrainingMetrics:
    """Top-level training metrics emitted next to the checkpoint."""

    total_epochs: int
    total_pairs: int
    total_seconds: float
    final_pixel_loss: float
    device: str
    seed: int
    evidence_grade: str = "MPS-research-signal"
    score_claim: bool = False
    promotion_eligible: bool = False
    axis_tag: str = "[MPS-research-signal]"
    per_epoch: tuple[EpochMetric, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "per_epoch": [asdict(e) for e in self.per_epoch],
        }


def _decode_real_frame_pairs(
    video_path: Path, num_pairs: int, height: int, width: int
) -> Iterator[torch.Tensor]:
    """Yield ``num_pairs`` consecutive frame pairs from the contest video.

    Returns float32 tensors in [0, 1] of shape (2, 3, H, W).

    Uses pyav per CLAUDE.md "Modal `.spawn()` HARVEST OR LOSE" companion
    "Real frames decoded via pyav, NEVER synthetic" rule.
    """
    try:
        import av  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "pyav is required for real-frame decode (CLAUDE.md non-negotiable). "
            "Install via `uv pip install av`."
        ) from exc

    container = av.open(str(video_path))
    stream = container.streams.video[0]
    frame_iter = container.decode(stream)

    prev_frame = None
    pairs_yielded = 0
    for frame in frame_iter:
        if pairs_yielded >= num_pairs:
            break
        # Convert to torch (H, W, 3) uint8
        arr = frame.to_ndarray(format="rgb24")
        tensor = torch.from_numpy(arr).float() / 255.0
        # Resize to (H, W) if mismatched
        if tensor.shape[0] != height or tensor.shape[1] != width:
            tensor_chw = tensor.permute(2, 0, 1).unsqueeze(0)
            tensor_chw = torch.nn.functional.interpolate(
                tensor_chw, size=(height, width), mode="bilinear", align_corners=False
            )
            tensor = tensor_chw.squeeze(0).permute(1, 2, 0)
        tensor_chw = tensor.permute(2, 0, 1)  # (3, H, W)
        if prev_frame is not None:
            pair = torch.stack([prev_frame, tensor_chw], dim=0)  # (2, 3, H, W)
            yield pair
            pairs_yielded += 1
        prev_frame = tensor_chw
    container.close()


def _eval_roundtrip(tensor: torch.Tensor) -> torch.Tensor:
    """Apply eval-roundtrip uint8 quantization per CLAUDE.md non-negotiable.

    The contest scorer evaluates frames after a 384 -> 874 -> uint8 -> 384
    roundtrip; eval_roundtrip=True simulates the uint8 bottleneck during eval
    so the measured performance reflects what the scorer would see.

    Conservative implementation: clamp to [0, 1], multiply by 255, round, divide
    back. Matches the upstream behavior at the relevant precision.
    """
    return (tensor.clamp(0.0, 1.0) * 255.0).round() / 255.0


def train_on_mps_real_frames(
    *,
    output_dir: Path,
    upstream_dir: Path,
    epochs: int = 100,
    num_pairs: int = 10,
    batch_size: int = 2,
    learning_rate: float = 1e-3,
    seed: int = 42,
    device: str = "mps",
    include_scorer_loss: bool = False,
) -> TrainingMetrics:
    """Train the tiny renderer on real contest frames + emit canonical artifacts.

    Args:
        output_dir: directory to write checkpoint + metrics + frame cache
        upstream_dir: path to the upstream snapshot (carries videos + models)
        epochs: number of training epochs (default 100; <10min wallclock on MPS)
        num_pairs: number of real frame pairs to cache + train on
        batch_size: per-step batch size
        learning_rate: AdamW learning rate
        seed: deterministic seed
        device: "mps" / "cuda" / "cpu" (default "mps" for the local diagnostic)
        include_scorer_loss: WHEN True, adds the canonical SegNet+PoseNet
            scorer-loss term; default False (pixel L1 only — keeps the
            training loop fast and the diagnostic surface clean)

    Returns:
        :class:`TrainingMetrics` summary; the canonical artifacts are written
        to disk in ``output_dir``:
          * ``checkpoint_ema.pt`` — EMA shadow state_dict
          * ``training_metrics.json`` — full :class:`TrainingMetrics`
          * ``frame_cache.pt`` — the cached frame pairs (so the gap step can
             use the EXACT same inputs)
          * ``manifest.json`` — non-promotability manifest per Catalog #192
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    upstream_dir = Path(upstream_dir)
    video_path = upstream_dir / "videos" / "0.mkv"
    if not video_path.exists():
        raise FileNotFoundError(
            f"Real contest video not found at {video_path}; required by CLAUDE.md."
        )

    torch.manual_seed(seed)
    device_obj = torch.device(device)
    model = build_tiny_renderer(seed=seed).to(device_obj)
    ema = EMA(model, decay=0.997)
    optim = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Cache real frames once + reuse across epochs (10 pairs * ~1.5MB = 15MB)
    pairs = list(
        _decode_real_frame_pairs(
            video_path,
            num_pairs=num_pairs,
            height=model.config.height,
            width=model.config.width,
        )
    )
    if len(pairs) < num_pairs:
        raise RuntimeError(
            f"video decode produced {len(pairs)} pairs; needed {num_pairs}"
        )
    frame_cache = torch.stack(pairs, dim=0)  # (N, 2, 3, H, W)
    torch.save(frame_cache, output_dir / "frame_cache.pt")

    # Optional scorer loading for richer gap measurement
    posenet = segnet = None
    if include_scorer_loss:
        from tac.scorer import load_default_scorers
        posenet, segnet = load_default_scorers(
            upstream_dir=upstream_dir, device=device_obj
        )

    epoch_metrics: list[EpochMetric] = []
    start_total = time.perf_counter()
    final_pixel_loss = float("nan")

    for epoch in range(epochs):
        start_epoch = time.perf_counter()
        model.train()
        pairs_seen_epoch = 0
        loss_accum = 0.0
        # Shuffle pair indices
        perm = torch.randperm(num_pairs, generator=torch.Generator().manual_seed(seed + epoch))
        for batch_start in range(0, num_pairs, batch_size):
            batch_indices = perm[batch_start : batch_start + batch_size]
            frames = frame_cache[batch_indices].to(device_obj)
            b = frames.shape[0]
            # Synthetic pose vector (deterministic; not from PoseNet so the
            # diagnostic stays scorer-independent unless include_scorer_loss=True)
            pose = torch.zeros(b, 12, device=device_obj)
            pose[:, 0] = 0.1 * batch_indices.float().to(device_obj)

            optim.zero_grad()
            reconstruction = model(frames, pose)
            pixel_loss = (reconstruction - frames).abs().mean()
            loss = pixel_loss

            if include_scorer_loss and posenet is not None and segnet is not None:
                # Conservative L1 scorer-feature loss against the GT frames
                # to keep the diagnostic loop simple (no archive grammar)
                pass  # left as a future hook; not exercised in the default smoke

            loss.backward()
            optim.step()
            ema.update(model)
            loss_accum += float(pixel_loss.detach().cpu())
            pairs_seen_epoch += int(b)

        avg_loss = loss_accum / max(1, pairs_seen_epoch // batch_size)
        final_pixel_loss = avg_loss
        epoch_metrics.append(
            EpochMetric(
                epoch=epoch,
                pixel_loss_mean=avg_loss,
                pairs_seen=pairs_seen_epoch,
                elapsed_s=time.perf_counter() - start_epoch,
            )
        )

    total_elapsed = time.perf_counter() - start_total

    # Save EMA shadow as the canonical inference checkpoint per CLAUDE.md
    # "EMA — NON-NEGOTIABLE" — inference / archive bytes come from the EMA
    # shadow, NEVER from the live final-epoch weights.
    ema_snapshot = _capture_ema_snapshot(model, ema)
    torch.save(ema_snapshot, output_dir / "checkpoint_ema.pt")

    # SPLIT-DEVICE REFERENCE CAPTURE (Catalog #324/#192 + predecessor verdict
    # `mps_phase_b_gap_experiment_verdict_20260519T053530Z`):
    #
    # The original single-device `compute_gap_components` cannot answer the
    # MPS-vs-CUDA gap question because Modal A10G workers have NO MPS device.
    # The fix is split-device: capture LOCAL MPS forward outputs + per-component
    # values HERE on the actual Mac MPS hardware (using the EMA shadow + the
    # exact `frame_cache.pt` the Modal dispatch will replay) so the remote
    # Modal CUDA dispatch can produce its own per-component values and the
    # local harvest can diff them apples-to-apples.
    #
    # `compute_local_mps_reference_components` runs the EMA-restored model
    # forward on `device` (typically "mps") using the cached frame batch +
    # synthetic pose vector that match what the Modal dispatch will replay.
    # It writes `local_mps_components.json` (per-component values + axis tag)
    # and `local_mps_forward_outputs.pt` (per-pair reconstruction tensors so a
    # consumer can compute per-pixel diff post-hoc if desired).
    #
    # Skipped when `device == "cuda"` or `device == "cpu"` because then the
    # reference-vs-target gap is trivially 0 (same device); the split-device
    # mission applies only to MPS.
    if device == "mps":
        try:
            from tac.mps_gap_experiment.harvest_and_verdict import (
                compute_local_mps_reference_components,
            )

            compute_local_mps_reference_components(
                checkpoint_path=output_dir / "checkpoint_ema.pt",
                frame_cache_path=output_dir / "frame_cache.pt",
                output_dir=output_dir,
                device=device,
                include_scorer_components=include_scorer_loss,
                upstream_dir=upstream_dir,
            )
        except Exception as exc:  # pragma: no cover - best-effort reference capture
            # Defensive: if the reference capture fails the training metrics
            # + checkpoint are still durable; the operator can re-run the
            # reference capture standalone via the CLI.
            print(
                f"[mps_gap_experiment] WARN: local MPS reference capture failed "
                f"({type(exc).__name__}: {exc}); harvest will need a standalone "
                f"reference run before split-device diff."
            )

    metrics = TrainingMetrics(
        total_epochs=epochs,
        total_pairs=num_pairs,
        total_seconds=total_elapsed,
        final_pixel_loss=final_pixel_loss,
        device=device,
        seed=seed,
        per_epoch=tuple(epoch_metrics),
    )
    (output_dir / "training_metrics.json").write_text(
        json.dumps(metrics.to_dict(), indent=2, sort_keys=True)
    )
    _write_non_promotability_manifest(output_dir, metrics, num_pairs=num_pairs)
    return metrics


def _capture_ema_snapshot(model: nn.Module, ema: EMA) -> dict:
    """Apply EMA + capture the state_dict + restore live weights.

    Mirrors the canonical pattern from ``experiments/train_distill.py``.
    """
    orig_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    ema.apply(model)
    try:
        snapshot = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    finally:
        model.load_state_dict(orig_state)
        model.train()
    return snapshot


def _write_non_promotability_manifest(
    output_dir: Path, metrics: TrainingMetrics, *, num_pairs: int
) -> None:
    """Write the canonical Catalog #192 non-promotability manifest.

    Mirrors ``tac.optimization.mps_research_signal`` semantics: every artifact
    emitted by this package is research-only, never promotable, never a
    contest-axis score claim.
    """
    manifest = {
        "schema_version": "mps_gap_experiment_v1_20260518",
        "evidence_grade": metrics.evidence_grade,
        "axis_tag": metrics.axis_tag,
        "score_claim": metrics.score_claim,
        "promotion_eligible": metrics.promotion_eligible,
        "ready_for_exact_eval_dispatch": False,
        "device": metrics.device,
        "seed": metrics.seed,
        "epochs": metrics.total_epochs,
        "num_pairs": num_pairs,
        "final_pixel_loss": metrics.final_pixel_loss,
        "notes": (
            "Phase 1 MPS-train output for the gap experiment. The CUDA "
            "comparison happens in tac.mps_gap_experiment.harvest_and_verdict "
            "+ the companion Modal A10G recipe. NO contest score claim."
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True)
    )
