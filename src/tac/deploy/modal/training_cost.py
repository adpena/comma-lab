"""Modal training cost-band anchor helpers.

This module keeps the cost posterior wire-in out of lane-local wrappers. Modal
training launchers record dispatch metadata, and recovery/harvest paths call
``append_modal_training_cost_anchor`` exactly once when a remote call returns.
"""

from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path
from typing import Any

from tac.cost_band_calibration import CostBandAnchor, append_anchor
from tac.repo_io import json_text

MODAL_GPU_HOURLY_RATES_USD = {
    "T4": 0.59,
    "A10G": 1.10,
    "A100": 4.00,
    "A100-40GB": 4.00,
    "A100-80GB": 4.00,
    "H100": 3.90,
    "H100-80GB": 3.90,
}


def normalize_modal_gpu(gpu: str) -> str:
    """Return a canonical Modal GPU label used by cost-band buckets."""

    value = str(gpu or "").strip().upper()
    if value == "A10G":
        return "A10G"
    if value.startswith("A100"):
        return value if value in MODAL_GPU_HOURLY_RATES_USD else "A100"
    if value.startswith("H100"):
        return value if value in MODAL_GPU_HOURLY_RATES_USD else "H100"
    if value == "T4":
        return "T4"
    return value


def estimate_modal_training_cost_usd(gpu: str, elapsed_seconds: float) -> tuple[float, float]:
    """Estimate provider cost from Modal GPU class and measured elapsed seconds.

    Modal result payloads expose wall-clock seconds, not invoice rows. This is
    therefore an estimate from a committed hourly-rate table, and callers must
    preserve the returned rate/source in the anchor notes.
    """

    gpu_norm = normalize_modal_gpu(gpu)
    rate = MODAL_GPU_HOURLY_RATES_USD.get(gpu_norm)
    if rate is None:
        raise ValueError(f"no Modal hourly rate configured for gpu={gpu!r}")
    elapsed = float(elapsed_seconds)
    if not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError(f"elapsed_seconds must be finite and nonnegative; got {elapsed!r}")
    return rate * elapsed / 3600.0, rate


def _utc_now() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat()


def _cost_band_metadata(metadata: dict[str, Any]) -> dict[str, Any] | None:
    payload = metadata.get("cost_band_anchor")
    return payload if isinstance(payload, dict) else None


def _bool_field(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _numeric_field(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return None


def append_modal_training_cost_anchor(
    *,
    out_dir: Path,
    metadata: dict[str, Any],
    result: dict[str, Any],
    posterior_path: Path | None = None,
    lock_path: Path | None = None,
) -> dict[str, Any]:
    """Append a Modal training cost anchor once, if launch metadata requests it.

    Returns a small status manifest and writes it to
    ``cost_band_anchor_appended.json`` in ``out_dir``. Re-running recovery is
    idempotent: an existing marker file is returned and no second posterior row
    is appended.
    """

    out_dir = Path(out_dir)
    marker = out_dir / "cost_band_anchor_appended.json"
    if marker.is_file():
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return {**payload, "already_appended": True}
        except json.JSONDecodeError:
            pass

    cost_meta = _cost_band_metadata(metadata)
    if cost_meta is None:
        return {
            "schema": "modal_training_cost_anchor_append_v1",
            "appended": False,
            "reason": "metadata_missing_cost_band_anchor",
        }

    elapsed = result.get("elapsed_seconds")
    if not isinstance(elapsed, (int, float)) or isinstance(elapsed, bool):
        return {
            "schema": "modal_training_cost_anchor_append_v1",
            "appended": False,
            "reason": "result_missing_numeric_elapsed_seconds",
        }

    gpu = normalize_modal_gpu(str(metadata.get("gpu") or cost_meta.get("gpu") or ""))
    try:
        estimated_cost, hourly_rate = estimate_modal_training_cost_usd(gpu, float(elapsed))
        epochs = int(cost_meta["epochs"])
        batch_size = int(cost_meta["batch_size"])
        trainer = str(cost_meta["trainer"])
    except (KeyError, TypeError, ValueError) as exc:
        return {
            "schema": "modal_training_cost_anchor_append_v1",
            "appended": False,
            "reason": f"invalid_cost_band_metadata:{type(exc).__name__}:{exc}",
        }

    label = str(metadata.get("label") or cost_meta.get("dispatch_label") or "modal_training")
    rc = result.get("returncode")
    timed_out = bool(result.get("timed_out", False))
    notes = (
        "cost_estimate_source=modal_elapsed_seconds_x_configured_hourly_rate; "
        f"hourly_rate_usd={hourly_rate}; returncode={rc}; timed_out={timed_out}"
    )
    if cost_meta.get("notes"):
        notes += f"; {cost_meta['notes']}"
    anchor = CostBandAnchor(
        logged_at_utc=_utc_now(),
        dispatch_label=label,
        trainer=trainer,
        platform="modal",
        gpu=gpu,
        epochs=epochs,
        batch_size=batch_size,
        all_flags_on=_bool_field(cost_meta.get("all_flags_on", False)),
        actual_wall_clock_sec=float(elapsed),
        actual_cost_usd=estimated_cost,
        predicted_cost_usd_low=cost_meta.get("predicted_cost_usd_low"),
        predicted_cost_usd_high=cost_meta.get("predicted_cost_usd_high"),
        prediction_in_band=(
            bool(predicted_low <= estimated_cost <= predicted_high)
            if (predicted_low := _numeric_field(cost_meta.get("predicted_cost_usd_low"))) is not None
            and (predicted_high := _numeric_field(cost_meta.get("predicted_cost_usd_high"))) is not None
            else None
        ),
        notes=notes,
    )
    append_anchor(anchor, posterior_path=posterior_path, lock_path=lock_path)
    manifest = {
        "schema": "modal_training_cost_anchor_append_v1",
        "appended": True,
        "already_appended": False,
        "score_claim": False,
        "promotion_eligible": False,
        "cost_estimate": True,
        "cost_estimate_source": "modal_elapsed_seconds_x_configured_hourly_rate",
        "dispatch_label": label,
        "trainer": trainer,
        "platform": "modal",
        "gpu": gpu,
        "epochs": epochs,
        "batch_size": batch_size,
        "all_flags_on": anchor.all_flags_on,
        "elapsed_seconds": float(elapsed),
        "estimated_cost_usd": estimated_cost,
        "hourly_rate_usd": hourly_rate,
        "posterior_path": str(posterior_path) if posterior_path is not None else None,
        "notes": notes,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    marker.write_text(json_text(manifest), encoding="utf-8")
    return manifest


__all__ = [
    "MODAL_GPU_HOURLY_RATES_USD",
    "append_modal_training_cost_anchor",
    "estimate_modal_training_cost_usd",
    "normalize_modal_gpu",
]
