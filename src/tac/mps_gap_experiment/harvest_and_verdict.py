# SPDX-License-Identifier: MIT
"""Compute the MPS-train CUDA-score gap + emit the verdict manifest.

Pipeline (Phase 4 of the build plan):

1. Load the EMA shadow checkpoint produced by
   :func:`tac.mps_gap_experiment.train_on_mps.train_on_mps_real_frames`.
2. Run forward on the *target* device (typically CUDA on Modal A10G; can be
   CPU locally for dry-run smoke).
3. Compute the canonical scorer components (SegNet + PoseNet) on the
   reconstructed pairs.
4. Diff the per-component values against an MPS-forward reference
   (computed locally on the same checkpoint + same frame cache).
5. Emit ``gap_results.json`` with per-component + aggregate gap +
   classification verdict.

Verdict thresholds (per the landing plan):

* ``gap_relative_aggregate < 5%`` → ``LOCAL_MPS_TRAIN_VIABLE``
* ``5% <= gap_relative_aggregate < 20%`` → ``LOCAL_MPS_TRAIN_VIABLE_ADVISORY_ONLY``
* ``gap_relative_aggregate >= 20%`` → ``LOCAL_MPS_TRAIN_NOT_VIABLE_PIVOT_MLX_OR_VTOOLBOX``

NOT a contest substrate. Every artifact tagged ``MPS-research-signal`` /
``diagnostic-CUDA Modal A10G`` per Catalog #192 / #317 / CLAUDE.md "MPS auth
eval is NOISE" non-negotiable.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from tac.mps_gap_experiment.tiny_renderer import (
    TinyRenderer,
    build_tiny_renderer,
)

__all__ = (
    "GapManifest",
    "classify_verdict",
    "compute_gap_components",
)


# Verdict thresholds — match the landing plan exactly.
_VIABLE_THRESHOLD = 0.05
_ADVISORY_THRESHOLD = 0.20


@dataclass(frozen=True)
class ComponentGap:
    """Per-scorer-component gap row."""

    name: str
    mps_value: float
    target_value: float
    absolute_diff: float
    relative_diff: float


@dataclass(frozen=True)
class GapManifest:
    """Top-level gap-experiment verdict manifest."""

    target_device: str
    mps_reference_device: str
    num_pairs: int
    verdict: str
    gap_relative_aggregate: float
    components: tuple[ComponentGap, ...] = field(default_factory=tuple)
    evidence_grade: str = "MPS-research-signal"
    score_claim: bool = False
    promotion_eligible: bool = False

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "components": [asdict(c) for c in self.components],
        }


def classify_verdict(gap_relative_aggregate: float) -> str:
    """Map an aggregate relative gap to the canonical verdict string."""
    if gap_relative_aggregate < _VIABLE_THRESHOLD:
        return "LOCAL_MPS_TRAIN_VIABLE"
    if gap_relative_aggregate < _ADVISORY_THRESHOLD:
        return "LOCAL_MPS_TRAIN_VIABLE_ADVISORY_ONLY"
    return "LOCAL_MPS_TRAIN_NOT_VIABLE_PIVOT_MLX_OR_VTOOLBOX"


def _eval_on_device(
    *,
    checkpoint_path: Path,
    frame_cache_path: Path,
    device: str,
    include_scorer_components: bool,
    upstream_dir: Path | None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Run forward + (optionally) scorer on a target device.

    Returns:
        (reconstruction_tensor, components_dict). The components_dict has
        keys like {"pixel_l1_mean", "pose_dist_mean", "seg_dist_mean"};
        scorer-derived keys only populated when ``include_scorer_components``
        is True and ``upstream_dir`` is provided.
    """
    device_obj = torch.device(device)
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = build_tiny_renderer().to(device_obj)
    model.load_state_dict(state_dict)
    model.eval()

    frame_cache = torch.load(frame_cache_path, map_location="cpu", weights_only=True)
    frame_cache = frame_cache.to(device_obj)
    n = frame_cache.shape[0]
    pose = torch.zeros(n, 12, device=device_obj)
    pose[:, 0] = 0.1 * torch.arange(n, device=device_obj).float()

    with torch.no_grad():
        reconstruction = model(frame_cache, pose)
        pixel_l1 = (reconstruction - frame_cache).abs().mean().item()

    components: dict[str, float] = {"pixel_l1_mean": pixel_l1}

    if include_scorer_components and upstream_dir is not None:
        from tac.scorer import load_default_scorers

        posenet, segnet = load_default_scorers(
            upstream_dir=upstream_dir, device=device_obj
        )

        # SegNet expects (B, T, 3, H, W) -> takes x[:, -1]; the scorer's
        # canonical preprocess is handled inside the helper but we go direct
        # here for the diagnostic surface.
        with torch.no_grad():
            try:
                seg_out = segnet(reconstruction[:, -1, ...])
                seg_value = float(seg_out.float().mean().item())
            except Exception:
                seg_value = float("nan")
            try:
                pose_out = posenet(reconstruction)
                pose_value = float(pose_out.float().mean().item())
            except Exception:
                pose_value = float("nan")
        components["segnet_mean_output"] = seg_value
        components["posenet_mean_output"] = pose_value

    return reconstruction.detach().cpu(), components


def compute_gap_components(
    *,
    checkpoint_path: Path,
    frame_cache_path: Path,
    output_path: Path,
    target_device: str = "cuda",
    mps_reference_device: str = "mps",
    include_scorer_components: bool = False,
    upstream_dir: Path | None = None,
) -> GapManifest:
    """Compute MPS-vs-target gap + write the canonical manifest.

    Args:
        checkpoint_path: ``checkpoint_ema.pt`` from
            :func:`train_on_mps_real_frames`
        frame_cache_path: ``frame_cache.pt`` from the same training run
            (CRITICAL: same inputs must be used on both devices)
        output_path: where to write the ``gap_results.json`` manifest
        target_device: the device we're comparing against MPS (typically
            "cuda" on Modal A10G; can be "cpu" for local dry-run)
        mps_reference_device: the MPS reference (always "mps" in practice)
        include_scorer_components: if True, load SegNet + PoseNet and include
            their per-output gap rows; default False
        upstream_dir: required if ``include_scorer_components=True``

    Returns:
        :class:`GapManifest` with verdict + per-component rows. The manifest
        is also written to ``output_path`` as JSON.
    """
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not Path(frame_cache_path).exists():
        raise FileNotFoundError(f"Frame cache not found: {frame_cache_path}")

    # MPS reference forward
    _, mps_components = _eval_on_device(
        checkpoint_path=Path(checkpoint_path),
        frame_cache_path=Path(frame_cache_path),
        device=mps_reference_device,
        include_scorer_components=include_scorer_components,
        upstream_dir=upstream_dir,
    )
    # Target device forward
    _, target_components = _eval_on_device(
        checkpoint_path=Path(checkpoint_path),
        frame_cache_path=Path(frame_cache_path),
        device=target_device,
        include_scorer_components=include_scorer_components,
        upstream_dir=upstream_dir,
    )

    component_rows: list[ComponentGap] = []
    rel_diffs: list[float] = []
    for key in sorted(mps_components.keys()):
        mps_val = mps_components[key]
        tgt_val = target_components.get(key, float("nan"))
        abs_diff = abs(mps_val - tgt_val)
        denom = max(abs(mps_val), 1e-12)
        rel_diff = abs_diff / denom
        component_rows.append(
            ComponentGap(
                name=key,
                mps_value=mps_val,
                target_value=tgt_val,
                absolute_diff=abs_diff,
                relative_diff=rel_diff,
            )
        )
        rel_diffs.append(rel_diff)

    gap_aggregate = sum(rel_diffs) / max(1, len(rel_diffs))
    verdict = classify_verdict(gap_aggregate)
    frame_cache = torch.load(frame_cache_path, map_location="cpu", weights_only=True)

    manifest = GapManifest(
        target_device=target_device,
        mps_reference_device=mps_reference_device,
        num_pairs=int(frame_cache.shape[0]),
        verdict=verdict,
        gap_relative_aggregate=gap_aggregate,
        components=tuple(component_rows),
    )
    Path(output_path).write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True)
    )
    return manifest
