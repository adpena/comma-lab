"""Adaptive-analyzer wrapper around the per-architecture-class profile registry.

This module is the **integration adapter** that lets the existing solvers
(cathedral_autopilot, meta-Lagrangian, theoretical_floor) consume per-class
calibration without changing their function signatures.

Backwards-compatibility contract
--------------------------------
- Solvers that don't pass an ``architecture_class`` get the HNeRV defaults
  (current behaviour). The registry's ``hnerv_ft_microcodec`` profile is the
  bootstrap source.
- Solvers that DO pass a class get per-class R values + decoder-aware
  decomposition + low-calibration band widening.

The split between this module and
``cuda_cpu_axis_profile_registry`` is deliberate: the registry holds the
DATA + bootstrap, this module holds the wiring. A solver should depend on
this module's stable API surface, not on the registry's internals.

Public API
----------
- :func:`get_registry`: lazy registry handle (loads from disk if present;
  bootstraps from HNeRV anchors otherwise)
- :func:`reload_registry`: force re-read (for cron-loop / harvest cycles)
- :func:`predict_cpu_score_band`: cathedral_autopilot adapter
- :func:`per_class_lagrangian_weights`: meta-Lagrangian adapter
- :func:`per_class_floor_band`: theoretical_floor adapter
- :func:`detect_class_from_payload`: thin wrapper over the classifier
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from tac.optimization.cuda_cpu_axis_profile_registry import (
    DEFAULT_DECODER_PROFILE,
    DEFAULT_REGISTRY_PATH,
    ArchitectureProfile,
    bootstrap_registry_from_hnerv_anchors,
    classify_archive_into_profile,
    confidence_aware_score_band,
    decompose_observed_drift,
    query_profile_for_archive_class,
    read_registry,
)

_REGISTRY_LOCK = threading.Lock()
_REGISTRY_CACHE: dict[str, ArchitectureProfile] | None = None
_REGISTRY_CACHE_PATH: Path | None = None


def get_registry(
    *,
    path: Path | str | None = None,
    force_reload: bool = False,
) -> dict[str, ArchitectureProfile]:
    """Return the cached registry; lazy-load from ``path`` on first call.

    Threadsafe via a module-level lock. Pass ``force_reload=True`` to
    invalidate the cache (e.g. after a harvest cycle wrote new anchors).
    """
    global _REGISTRY_CACHE, _REGISTRY_CACHE_PATH

    target_path = Path(path) if path is not None else DEFAULT_REGISTRY_PATH

    with _REGISTRY_LOCK:
        if (
            _REGISTRY_CACHE is None
            or force_reload
            or target_path != _REGISTRY_CACHE_PATH
        ):
            if target_path.exists():
                _REGISTRY_CACHE = read_registry(
                    target_path, bootstrap_if_missing=False
                )
            else:
                _REGISTRY_CACHE = bootstrap_registry_from_hnerv_anchors()
            _REGISTRY_CACHE_PATH = target_path
        return _REGISTRY_CACHE


def reload_registry(path: Path | str | None = None) -> dict[str, ArchitectureProfile]:
    """Force-reload the registry from disk."""
    return get_registry(path=path, force_reload=True)


# ── cathedral_autopilot adapter ────────────────────────────────────────────
def predict_cpu_score_band(
    *,
    cuda_score: float,
    architecture_class: str | None = None,
    archive_metadata: dict[str, Any] | None = None,
    archive_path: str | Path | None = None,
    registry: dict[str, ArchitectureProfile] | None = None,
) -> dict[str, Any]:
    """Predict the CPU contest score from a CUDA score, per architecture class.

    The caller may pass ``architecture_class`` explicitly; if not, this
    function classifies the archive via metadata + bytes. Falls back to
    ``unknown_uncalibrated`` (HNeRV defaults) when nothing matches.

    Returns the same dict shape as
    :func:`tac.optimization.cuda_cpu_axis_profile_registry.confidence_aware_score_band`,
    so cathedral_autopilot can drop it into a plan row directly.
    """
    if registry is None:
        registry = get_registry()

    if architecture_class is None:
        architecture_class = classify_archive_into_profile(
            archive_path=archive_path,
            archive_metadata=archive_metadata,
        )

    return confidence_aware_score_band(
        architecture_class=architecture_class,
        cuda_score=cuda_score,
        registry=registry,
    )


# ── meta-Lagrangian adapter ────────────────────────────────────────────────
def per_class_lagrangian_weights(
    architecture_class: str | None = None,
    *,
    registry: dict[str, ArchitectureProfile] | None = None,
    base_lambda_pose: float = 1.0,
    base_lambda_seg: float = 1.0,
    base_lambda_rate: float = 1.0,
) -> dict[str, float]:
    """Return per-axis Lagrangian multipliers calibrated for an architecture class.

    The intuition: for HNeRV at the medal band, paired CPU/CUDA rows show that
    CUDA-axis pose changes have lower expected marginal value on the CPU axis.
    This is a planning prior, not proof that pose work is useless. The
    CPU-axis weighting is::

        lambda_pose_eff = base_lambda_pose / R_pose
        lambda_seg_eff  = base_lambda_seg / R_seg
        lambda_rate_eff = base_lambda_rate    (rate is device-agnostic)

    Different architecture classes have different ``R`` values — the
    registry posterior provides the calibrated factor.

    Returns a dict with keys ``lambda_pose``, ``lambda_seg``, ``lambda_rate``,
    ``r_pose_used``, ``r_seg_used``, ``architecture_class_used``,
    ``confidence_label``.
    """
    if registry is None:
        registry = get_registry()

    cls = architecture_class or "hnerv_ft_microcodec"
    profile = query_profile_for_archive_class(cls, registry=registry)

    return {
        "lambda_pose": base_lambda_pose / profile.r_pose_mean,
        "lambda_seg": base_lambda_seg / profile.r_seg_mean,
        "lambda_rate": base_lambda_rate,
        "r_pose_used": profile.r_pose_mean,
        "r_seg_used": profile.r_seg_mean,
        "architecture_class_used": profile.architecture_class,
        "confidence_label": profile.confidence_label(),
        "evidence_grade": "[CPU-prep planning-only; per-class Lagrangian]",
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


# ── theoretical_floor adapter ──────────────────────────────────────────────
def per_class_floor_band(
    *,
    architecture_class: str | None = None,
    registry: dict[str, ArchitectureProfile] | None = None,
    cuda_pose_floor: float | None = None,
) -> dict[str, Any]:
    """Per-class CPU-axis floor band.

    Different architectures may have different observed CPU/CUDA pose bands.
    HNeRV currently uses an empirical paired-anchor prior. Balle hyperprior,
    AV1 grayscale, and other families are not yet measured and therefore carry
    low-calibration confidence.

    Returns ``{"cuda_pose_floor", "cpu_pose_floor_implied", "confidence_label",
    "architecture_class_used"}``. The CPU pose floor is implied via
    ``cpu_pose_floor = cuda_pose_floor / r_pose_mean`` as a reporting prior.
    """
    if registry is None:
        registry = get_registry()

    cls = architecture_class or "hnerv_ft_microcodec"
    profile = query_profile_for_archive_class(cls, registry=registry)
    floor = cuda_pose_floor if cuda_pose_floor is not None else profile.pose_floor_estimate

    return {
        "cuda_pose_floor": float(floor),
        "cpu_pose_floor_implied": float(floor) / profile.r_pose_mean,
        "r_pose_used": profile.r_pose_mean,
        "architecture_class_used": profile.architecture_class,
        "confidence_label": profile.confidence_label(),
        "evidence_grade": "[CPU-prep planning-only; per-class precision floor]",
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


# ── Decoder-network decomposition adapter ──────────────────────────────────
def explain_observed_pose_drift(
    *,
    observed_r_pose: float,
    architecture_class: str | None = None,
    registry: dict[str, ArchitectureProfile] | None = None,
) -> dict[str, Any]:
    """Combine the registry's profile + decoder split into a single explanation.

    Useful for audit reports and operator briefings: given an observed
    R_pose (e.g. 5.04 for HNeRV cluster), explain how much is decoder-side
    vs network-side.
    """
    if registry is None:
        registry = get_registry()

    cls = architecture_class or "hnerv_ft_microcodec"
    profile = query_profile_for_archive_class(cls, registry=registry)

    # Use the profile's decoder fraction as a mechanism prior — it falls back
    # to the default 25% hypothesis when no measured override is recorded.
    decoder_profile = DEFAULT_DECODER_PROFILE
    if profile.decoder_drift_fraction != DEFAULT_DECODER_PROFILE.pose_drift_fraction:
        # Profile has overridden the default — instantiate a per-class profile
        from tac.optimization.cuda_cpu_axis_profile_registry import DecoderProfile
        decoder_profile = DecoderProfile(
            decoder_pair=DEFAULT_DECODER_PROFILE.decoder_pair,
            pose_drift_fraction=profile.decoder_drift_fraction,
            seg_drift_fraction=DEFAULT_DECODER_PROFILE.seg_drift_fraction,
        )

    decomposition = decompose_observed_drift(
        observed_r_pose=observed_r_pose,
        decoder_profile=decoder_profile,
        pose_floor_estimate=profile.pose_floor_estimate,
    )
    decomposition["architecture_class_used"] = profile.architecture_class
    decomposition["confidence_label"] = profile.confidence_label()
    decomposition["n_anchors_backing"] = profile.n_anchors
    decomposition["score_claim"] = False
    decomposition["promotion_eligible"] = False
    decomposition["rank_or_kill_eligible"] = False
    decomposition["ready_for_exact_eval_dispatch"] = False
    return decomposition


# ── Detect-from-payload thin wrapper ───────────────────────────────────────
def detect_class_from_payload(payload: dict[str, Any]) -> str:
    """Thin wrapper that classifies a payload (e.g. PR scorecard row)."""
    return classify_archive_into_profile(archive_metadata=payload)


__all__ = [
    "detect_class_from_payload",
    "explain_observed_pose_drift",
    "get_registry",
    "per_class_floor_band",
    "per_class_lagrangian_weights",
    "predict_cpu_score_band",
    "reload_registry",
]
