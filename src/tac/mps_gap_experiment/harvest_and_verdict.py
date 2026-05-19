# SPDX-License-Identifier: MIT
"""Split-device gap-experiment harvest + verdict helpers.

**SPLIT-DEVICE ARCHITECTURE** (per predecessor verdict
`mps_phase_b_gap_experiment_verdict_20260519T053530Z` Option A reactivation):

The original single-machine ``compute_gap_components`` could NOT answer the
gating question "do MPS-trained weights survive CUDA scoring within usable
tolerance?" because Modal A10G workers have NO MPS hardware — both the
"reference" and "target" forwards ran on CUDA and produced identical outputs
(measurement artifact). The dispositive measurement requires THREE phases on
TWO machines:

1. **LOCAL (Mac MPS)** — :func:`compute_local_mps_reference_components` runs
   the EMA-restored model on Apple Silicon MPS using the cached frame batch
   and emits ``local_mps_components.json`` + ``local_mps_forward_outputs.pt``.
   Called as a post-training step inside
   :func:`tac.mps_gap_experiment.train_on_mps.train_on_mps_real_frames` so the
   reference is captured on the SAME machine + SAME EMA shadow + SAME input
   batch the Modal dispatch will replay.

2. **REMOTE (Modal A10G CUDA)** —
   :func:`compute_target_cuda_components` runs on the Modal worker against the
   uploaded EMA checkpoint + frame cache and emits
   ``target_cuda_components.json`` + ``target_cuda_forward_outputs.pt`` to the
   Modal output directory. Harvested locally via
   ``tools/harvest_modal_calls.py``.

3. **LOCAL (Mac)** — :func:`diff_components_and_classify_verdict` loads BOTH
   JSONs (NEVER recomputes on a single device — the gap is the diff of two
   pre-captured artifacts), computes per-component absolute + relative diff,
   applies the canonical verdict thresholds, and emits the canonical
   ``gap_results.json`` manifest.

Verdict thresholds (per the landing plan + recipe):

* ``gap_relative_aggregate < 5%`` → ``LOCAL_MPS_TRAIN_VIABLE``
* ``5% <= gap_relative_aggregate < 20%`` → ``LOCAL_MPS_TRAIN_VIABLE_ADVISORY_ONLY``
* ``gap_relative_aggregate >= 20%`` → ``LOCAL_MPS_TRAIN_NOT_VIABLE_PIVOT_MLX_OR_VTOOLBOX``

NOT a contest substrate. Every artifact tagged ``MPS-research-signal`` /
``diagnostic-CUDA Modal A10G`` per Catalog #192 / #317 / CLAUDE.md "MPS auth
eval is NOISE" non-negotiable.

Backward compatibility: :func:`compute_gap_components` is preserved as a
local-only dry-run self-comparison helper (gap is trivially 0 when target ==
reference). It MUST NOT be invoked on a Modal worker for a real MPS-vs-CUDA
verdict — that's what the split-device helpers are for.
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
    "ComponentGap",
    "GapManifest",
    "classify_verdict",
    "compute_local_mps_reference_components",
    "compute_target_cuda_components",
    "diff_components_and_classify_verdict",
    "compute_gap_components",
)


# Verdict thresholds — match the landing plan exactly.
_VIABLE_THRESHOLD = 0.05
_ADVISORY_THRESHOLD = 0.20

# Canonical evidence-grade strings per Catalog #192/#317 + CLAUDE.md
# "MPS auth eval is NOISE" non-negotiable. Both axes are non-promotable.
_LOCAL_MPS_AXIS_TAG = "[MPS-research-signal]"
_TARGET_CUDA_AXIS_TAG = "[diagnostic-CUDA Modal A10G]"
_LOCAL_MPS_EVIDENCE_GRADE = "MPS-research-signal"
_TARGET_CUDA_EVIDENCE_GRADE = "diagnostic-CUDA Modal A10G"

# Canonical filenames for the split-device artifacts. Keep in sync with the
# Modal dispatch driver + harness so harvest can find them by convention.
_LOCAL_MPS_COMPONENTS_FILENAME = "local_mps_components.json"
_LOCAL_MPS_FORWARD_OUTPUTS_FILENAME = "local_mps_forward_outputs.pt"
_TARGET_CUDA_COMPONENTS_FILENAME = "target_cuda_components.json"
_TARGET_CUDA_FORWARD_OUTPUTS_FILENAME = "target_cuda_forward_outputs.pt"


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
    # NaN propagates to the NOT_VIABLE bucket per the NaN-fallback semantics
    # the predecessor's harvest established.
    if gap_relative_aggregate != gap_relative_aggregate:  # NaN check
        return "LOCAL_MPS_TRAIN_NOT_VIABLE_PIVOT_MLX_OR_VTOOLBOX"
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

        # Catalog #222: scorer-loader assignment order is (posenet, segnet).
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


def _write_components_artifact(
    *,
    output_dir: Path,
    components_filename: str,
    forward_outputs_filename: str,
    device: str,
    axis_tag: str,
    evidence_grade: str,
    components: dict[str, float],
    reconstruction: torch.Tensor,
    schema_version: str,
    notes: str,
    extra: dict | None = None,
) -> tuple[Path, Path]:
    """Persist the per-side components JSON + forward-outputs tensor.

    Canonical artifact contract used by BOTH local MPS (via
    :func:`compute_local_mps_reference_components`) and remote CUDA (via
    :func:`compute_target_cuda_components`). The schema is intentionally
    permissive so :func:`diff_components_and_classify_verdict` can load either
    side without device-specific branches.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    components_path = output_dir / components_filename
    forward_outputs_path = output_dir / forward_outputs_filename

    manifest = {
        "schema_version": schema_version,
        "evidence_grade": evidence_grade,
        "axis_tag": axis_tag,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "device": device,
        "components": components,
        "num_pairs": int(reconstruction.shape[0]) if reconstruction.dim() >= 1 else 0,
        "forward_outputs_relpath": forward_outputs_filename,
        "notes": notes,
    }
    if extra:
        # Caller-supplied extra metadata (e.g. modal_call_id when known) is
        # merged shallowly without overwriting canonical keys.
        for key, value in extra.items():
            manifest.setdefault(key, value)

    components_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    torch.save(reconstruction.detach().cpu(), forward_outputs_path)
    return components_path, forward_outputs_path


def compute_local_mps_reference_components(
    *,
    checkpoint_path: Path,
    frame_cache_path: Path,
    output_dir: Path,
    device: str = "mps",
    include_scorer_components: bool = False,
    upstream_dir: Path | None = None,
) -> Path:
    """Capture the LOCAL MPS reference components + forward outputs.

    Runs on Apple Silicon Mac MPS hardware (NEVER on Modal). Writes:
      * ``{output_dir}/local_mps_components.json`` — per-component values
        tagged ``[MPS-research-signal]``
      * ``{output_dir}/local_mps_forward_outputs.pt`` — per-pair reconstruction
        tensor (CPU-resident; downstream consumers may load on any device).

    Per CLAUDE.md "MPS auth eval is NOISE" + Catalog #1 + Catalog #192 +
    Catalog #317: every artifact this helper writes is research-signal-only,
    never promotable, never a contest-axis score claim.

    Returns:
        Path to the written ``local_mps_components.json``.
    """
    if device != "mps":
        # Sister callers may invoke with "cpu" for unit tests; in that case
        # the axis tag still claims MPS-research-signal because the artifact
        # contract IS the reference side; the actual divergence-from-CUDA
        # surfaces in diff_components_and_classify_verdict. The non-MPS path
        # is research-only by construction (gap with CUDA reference equals 0).
        pass

    reconstruction, components = _eval_on_device(
        checkpoint_path=Path(checkpoint_path),
        frame_cache_path=Path(frame_cache_path),
        device=device,
        include_scorer_components=include_scorer_components,
        upstream_dir=upstream_dir,
    )
    components_path, _ = _write_components_artifact(
        output_dir=Path(output_dir),
        components_filename=_LOCAL_MPS_COMPONENTS_FILENAME,
        forward_outputs_filename=_LOCAL_MPS_FORWARD_OUTPUTS_FILENAME,
        device=device,
        axis_tag=_LOCAL_MPS_AXIS_TAG,
        evidence_grade=_LOCAL_MPS_EVIDENCE_GRADE,
        components=components,
        reconstruction=reconstruction,
        schema_version="mps_gap_local_mps_reference_v1_20260519",
        notes=(
            "Local MPS reference forward components captured on Apple Silicon "
            "MPS hardware against the EMA shadow + frame_cache.pt batch. "
            "Paired with target_cuda_components.json from the Modal A10G dispatch "
            "via tac.mps_gap_experiment.harvest_and_verdict.diff_components_and_"
            "classify_verdict."
        ),
    )
    return components_path


def compute_target_cuda_components(
    *,
    checkpoint_path: Path,
    frame_cache_path: Path,
    output_dir: Path,
    device: str = "cuda",
    include_scorer_components: bool = False,
    upstream_dir: Path | None = None,
    modal_call_id: str | None = None,
) -> Path:
    """Capture the REMOTE Modal CUDA (or local CUDA) target components.

    Runs on a Modal A10G worker (or any CUDA-capable host) against the
    uploaded EMA checkpoint + frame cache. Writes:
      * ``{output_dir}/target_cuda_components.json`` tagged
        ``[diagnostic-CUDA Modal A10G]``
      * ``{output_dir}/target_cuda_forward_outputs.pt``

    Per CLAUDE.md "Apples-to-apples evidence discipline": this is the target
    side of the split-device gap; the per-axis tag stays NON-promotable even
    though CUDA is "the better device" — the gap-experiment recipe is a
    diagnostic, not a contest score claim.

    Returns:
        Path to the written ``target_cuda_components.json``.
    """
    reconstruction, components = _eval_on_device(
        checkpoint_path=Path(checkpoint_path),
        frame_cache_path=Path(frame_cache_path),
        device=device,
        include_scorer_components=include_scorer_components,
        upstream_dir=upstream_dir,
    )
    extra: dict[str, object] = {}
    if modal_call_id:
        extra["modal_call_id"] = modal_call_id
    components_path, _ = _write_components_artifact(
        output_dir=Path(output_dir),
        components_filename=_TARGET_CUDA_COMPONENTS_FILENAME,
        forward_outputs_filename=_TARGET_CUDA_FORWARD_OUTPUTS_FILENAME,
        device=device,
        axis_tag=_TARGET_CUDA_AXIS_TAG,
        evidence_grade=_TARGET_CUDA_EVIDENCE_GRADE,
        components=components,
        reconstruction=reconstruction,
        schema_version="mps_gap_target_cuda_v1_20260519",
        notes=(
            "Modal A10G target-device forward components captured against the "
            "uploaded EMA shadow + frame_cache.pt batch. Paired with "
            "local_mps_components.json from the LOCAL MPS training step via "
            "tac.mps_gap_experiment.harvest_and_verdict.diff_components_and_"
            "classify_verdict."
        ),
        extra=extra,
    )
    return components_path


def _diff_component_dicts(
    *,
    mps_components: dict[str, float],
    target_components: dict[str, float],
) -> tuple[list[ComponentGap], float]:
    """Build the per-component gap rows + aggregate relative gap.

    Pure helper (no I/O); both callers (diff_components_and_classify_verdict
    + legacy compute_gap_components) share it so the verdict math is identical
    on both code paths.
    """
    component_rows: list[ComponentGap] = []
    rel_diffs: list[float] = []
    for key in sorted(mps_components.keys()):
        mps_val = float(mps_components[key])
        tgt_val = float(target_components.get(key, float("nan")))
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
    return component_rows, gap_aggregate


def diff_components_and_classify_verdict(
    *,
    local_mps_components_path: Path,
    target_cuda_components_path: Path,
    output_path: Path,
) -> GapManifest:
    """Diff a pre-captured LOCAL MPS + REMOTE CUDA components pair.

    This is the canonical split-device harvest helper — it does NOT run any
    forward pass on either device. It loads two JSON artifacts produced by
    :func:`compute_local_mps_reference_components` (locally) and
    :func:`compute_target_cuda_components` (on Modal A10G), computes the
    per-component diff, applies the canonical verdict thresholds, and writes
    the canonical ``gap_results.json`` manifest.

    Per CLAUDE.md "Apples-to-apples evidence discipline": both sides MUST
    have been produced from the SAME EMA checkpoint and the SAME
    ``frame_cache.pt`` batch — otherwise the gap is a meaningless multi-axis
    drift number, not the MPS-vs-CUDA gap the mission asked for. The two
    component JSONs persist their `device` field + `num_pairs` field so the
    helper can sanity-check the pairing.

    Args:
        local_mps_components_path: path to ``local_mps_components.json``
            written by :func:`compute_local_mps_reference_components`
        target_cuda_components_path: path to ``target_cuda_components.json``
            written by :func:`compute_target_cuda_components` (typically
            harvested from the Modal output directory)
        output_path: where to write the canonical ``gap_results.json``

    Returns:
        :class:`GapManifest` with verdict + per-component rows. The manifest
        is also written to ``output_path`` as JSON.
    """
    local_path = Path(local_mps_components_path)
    target_path = Path(target_cuda_components_path)
    if not local_path.exists():
        raise FileNotFoundError(
            f"local MPS components not found: {local_path} (run the LOCAL "
            f"training step or compute_local_mps_reference_components first)"
        )
    if not target_path.exists():
        raise FileNotFoundError(
            f"target CUDA components not found: {target_path} (run the Modal "
            f"A10G dispatch + harvest first)"
        )

    local_manifest = json.loads(local_path.read_text())
    target_manifest = json.loads(target_path.read_text())

    mps_components = local_manifest.get("components", {})
    target_components = target_manifest.get("components", {})

    # Sanity: num_pairs MUST agree across the two sides (same frame_cache.pt
    # batch). If they disagree the comparison is meaningless; the helper
    # raises rather than emit a misleading verdict.
    local_num_pairs = int(local_manifest.get("num_pairs", 0))
    target_num_pairs = int(target_manifest.get("num_pairs", 0))
    if local_num_pairs != target_num_pairs:
        raise ValueError(
            f"num_pairs mismatch between local ({local_num_pairs}) and target "
            f"({target_num_pairs}); the two sides must compare the SAME "
            f"frame_cache.pt batch"
        )

    component_rows, gap_aggregate = _diff_component_dicts(
        mps_components=mps_components, target_components=target_components
    )
    verdict = classify_verdict(gap_aggregate)

    manifest = GapManifest(
        target_device=str(target_manifest.get("device", "cuda")),
        mps_reference_device=str(local_manifest.get("device", "mps")),
        num_pairs=local_num_pairs,
        verdict=verdict,
        gap_relative_aggregate=gap_aggregate,
        components=tuple(component_rows),
    )
    Path(output_path).write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True)
    )
    return manifest


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
    """LOCAL-ONLY dry-run self-comparison helper (backward compatible).

    .. warning::

       This helper runs BOTH forwards on the SAME host. It CANNOT answer the
       real MPS-vs-CUDA gap question — that's what
       :func:`diff_components_and_classify_verdict` is for. The Modal-side
       dispatch MUST use :func:`compute_target_cuda_components` (NOT this
       helper) so the LOCAL MPS reference is captured on actual Mac hardware
       and the gap is the diff of two pre-captured artifacts.

       Predecessor verdict
       ``mps_phase_b_gap_experiment_verdict_20260519T053530Z`` documents the
       single-device measurement artifact this helper produces when invoked
       on a Modal worker (no MPS hardware → both forwards run on CUDA → gap
       trivially 0.0).

    Preserved for backward compatibility with:
      * the existing test
        ``test_compute_gap_components_cpu_self_comparison_returns_zero_gap``
      * any local dry-run smoke that wants a self-consistent gap-math
        primitive check.
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

    component_rows, gap_aggregate = _diff_component_dicts(
        mps_components=mps_components, target_components=target_components
    )
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
