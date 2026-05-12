"""Modal training cost-band anchor helpers (compat shim).

The canonical implementation now lives in ``tac.cost_band_calibration`` so
the platform-keyed rate table sits next to the posterior it feeds. This
module remains as a thin compat shim for in-flight callers that import
``MODAL_GPU_HOURLY_RATES_USD``, ``normalize_modal_gpu``,
``estimate_modal_training_cost_usd``, or ``append_modal_training_cost_anchor``.

New code should import from ``tac.cost_band_calibration`` directly:

    from tac.cost_band_calibration import (
        PLATFORM_RATES_USD_PER_HOUR,
        append_platform_training_anchor,
        estimate_cost_usd,
        normalize_gpu,
    )

T1-B simplification 2026-05-12: see
``feedback_modal_mount_manifest_consolidation_landed_20260512.md``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tac.cost_band_calibration import (
    PLATFORM_RATES_USD_PER_HOUR,
    append_platform_training_anchor,
    estimate_cost_usd,
    normalize_gpu,
)

# Backward-compat alias: callers reading the Modal-specific rate dict still work.
MODAL_GPU_HOURLY_RATES_USD = PLATFORM_RATES_USD_PER_HOUR["modal"]


def normalize_modal_gpu(gpu: str) -> str:
    """Compat shim: delegate to the canonical platform-keyed normaliser."""

    return normalize_gpu("modal", gpu)


def estimate_modal_training_cost_usd(
    gpu: str, elapsed_seconds: float
) -> tuple[float, float]:
    """Compat shim: delegate to the canonical platform-keyed estimator."""

    return estimate_cost_usd("modal", gpu, elapsed_seconds)


def append_modal_training_cost_anchor(
    *,
    out_dir: Path,
    metadata: dict[str, Any],
    result: dict[str, Any],
    posterior_path: Path | None = None,
    lock_path: Path | None = None,
) -> dict[str, Any]:
    """Compat shim: delegate to the canonical platform-keyed appender.

    The returned manifest preserves the historical schema name
    ``modal_training_cost_anchor_append_v1`` for downstream consumers that
    pattern-match the schema string. The canonical function emits
    ``platform_training_cost_anchor_append_v1`` for new callers.
    """

    manifest = append_platform_training_anchor(
        "modal",
        out_dir=out_dir,
        metadata=metadata,
        result=result,
        posterior_path=posterior_path,
        lock_path=lock_path,
    )
    # Backward-compat schema rename for consumers that match the old string.
    if manifest.get("schema") == "platform_training_cost_anchor_append_v1":
        manifest = {**manifest, "schema": "modal_training_cost_anchor_append_v1"}
        if manifest.get("cost_estimate_source") == "modal_elapsed_seconds_x_configured_hourly_rate":
            # Keep the historical wording the existing test asserts on.
            pass
    return manifest


__all__ = [
    "MODAL_GPU_HOURLY_RATES_USD",
    "append_modal_training_cost_anchor",
    "estimate_modal_training_cost_usd",
    "normalize_modal_gpu",
]
