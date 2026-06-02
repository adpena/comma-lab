# SPDX-License-Identifier: MIT
"""Fail-closed contract for a native-MLX SNeRV train/export/archive adapter.

The current SNeRV execution path is a real receiver-priced CPU advisory lane.
It is useful, but it is not the native MLX train/export/archive adapter required
for long campaigns. This module makes that missing surface machine-checkable so
the runner can discover a real adapter when it lands and refuse fake execution
until the required symbols and signatures exist.
"""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

SNERV_MLX_NATIVE_ADAPTER_CONTRACT_SCHEMA = "snerv_mlx_native_adapter_contract.v1"
DEFAULT_SNERV_MLX_NATIVE_ADAPTER_MODULE = (
    "tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export"
)

FALSE_AUTHORITY = {
    "score_claim": False,
    "frontier_score_claim": False,
    "rank_or_kill_eligible": False,
    "promotion_eligible": False,
    "ready_for_exact_eval_dispatch": False,
}


@dataclass(frozen=True)
class RequiredSurface:
    """One required callable surface for the native adapter."""

    surface_id: str
    symbol: str
    required_parameters: tuple[str, ...]
    rationale: str


REQUIRED_SURFACES: tuple[RequiredSurface, ...] = (
    RequiredSurface(
        surface_id="train_export",
        symbol="train_export_snerv_mlx_native",
        required_parameters=(
            "output_dir",
            "num_pairs",
            "source_video_path",
            "modelsize_candidate",
            "scorer_upstream_dir",
        ),
        rationale=(
            "native MLX training must bind selected modelsize bytes, real video, "
            "and scorer custody in the same callable"
        ),
    ),
    RequiredSurface(
        surface_id="archive_export",
        symbol="export_snerv_mlx_archive",
        required_parameters=("model_or_artifact", "output_dir", "repo_root"),
        rationale=(
            "training is not useful until it emits a byte-closed contest archive"
        ),
    ),
    RequiredSurface(
        surface_id="receiver_proof",
        symbol="write_snerv_mlx_receiver_proof",
        required_parameters=("archive_zip_path", "runtime_submission_dir", "output_dir"),
        rationale="every native archive must prove inflate/runtime consumption",
    ),
    RequiredSurface(
        surface_id="local_mlx_prefilter",
        symbol="write_snerv_mlx_prefilter_profile",
        required_parameters=(
            "artifact",
            "archive_bytes",
            "archive_sha256",
            "output_path",
            "upstream_dir",
        ),
        rationale=(
            "CPU replay and exact auth are gated by full-video local MLX "
            "acquisition evidence"
        ),
    ),
)


def _signature_accepts_required_parameters(
    fn: Any,
    required_parameters: tuple[str, ...],
) -> tuple[bool, list[str], str | None]:
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError) as exc:
        return False, list(required_parameters), f"signature_unreadable:{exc!s}"
    params = signature.parameters
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()):
        return True, [], None
    missing = [name for name in required_parameters if name not in params]
    return not missing, missing, None


def build_snerv_mlx_native_adapter_contract(
    *,
    module_name: str = DEFAULT_SNERV_MLX_NATIVE_ADAPTER_MODULE,
    extra_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return fail-closed native-MLX adapter readiness for SNeRV."""

    blockers: list[str] = []
    surface_rows: list[dict[str, Any]] = []
    module_loaded = False
    module_error: str | None = None
    module: Any | None = None
    try:
        module = importlib.import_module(module_name)
        module_loaded = True
    except Exception as exc:  # pragma: no cover - exact exception is environment-specific
        module_error = repr(exc)
        blockers.append("snerv_mlx_native_train_export_archive_adapter_missing")

    for surface in REQUIRED_SURFACES:
        symbol_present = module_loaded and hasattr(module, surface.symbol)
        callable_present = False
        signature_ok = False
        missing_parameters: list[str] = list(surface.required_parameters)
        signature_error: str | None = None
        if symbol_present:
            value = getattr(module, surface.symbol)
            callable_present = callable(value)
            if callable_present:
                signature_ok, missing_parameters, signature_error = (
                    _signature_accepts_required_parameters(
                        value,
                        surface.required_parameters,
                    )
                )
        if not symbol_present:
            blockers.append(f"snerv_mlx_native_surface_missing:{surface.surface_id}")
        elif not callable_present:
            blockers.append(
                f"snerv_mlx_native_surface_not_callable:{surface.surface_id}"
            )
        elif not signature_ok:
            blockers.append(
                f"snerv_mlx_native_surface_signature_mismatch:{surface.surface_id}"
            )
        surface_rows.append(
            {
                "schema": "snerv_mlx_native_adapter_surface_row.v1",
                "surface_id": surface.surface_id,
                "symbol": surface.symbol,
                "symbol_present": bool(symbol_present),
                "callable": bool(callable_present),
                "signature_ok": bool(signature_ok),
                "required_parameters": list(surface.required_parameters),
                "missing_parameters": missing_parameters,
                "signature_error": signature_error,
                "rationale": surface.rationale,
            }
        )

    surfaces_ready = module_loaded and not blockers
    evidence = dict(extra_evidence or {})
    if surfaces_ready and not bool(evidence.get("two_pair_smoke_passed")):
        blockers.append("snerv_mlx_native_adapter_surfaces_present_but_unproven")

    return {
        "schema": SNERV_MLX_NATIVE_ADAPTER_CONTRACT_SCHEMA,
        "module_name": str(module_name),
        "module_loaded": bool(module_loaded),
        "module_error": module_error,
        "required_surface_count": len(REQUIRED_SURFACES),
        "ready_surface_count": sum(
            1
            for row in surface_rows
            if row["symbol_present"] and row["callable"] and row["signature_ok"]
        ),
        "surfaces_ready": bool(surfaces_ready),
        "two_pair_smoke_passed": bool(evidence.get("two_pair_smoke_passed")),
        "full600_campaign_ready": bool(
            surfaces_ready and evidence.get("two_pair_smoke_passed")
        ),
        "surface_rows": surface_rows,
        "blockers": list(dict.fromkeys(blockers)),
        **FALSE_AUTHORITY,
    }


__all__ = [
    "DEFAULT_SNERV_MLX_NATIVE_ADAPTER_MODULE",
    "REQUIRED_SURFACES",
    "SNERV_MLX_NATIVE_ADAPTER_CONTRACT_SCHEMA",
    "RequiredSurface",
    "build_snerv_mlx_native_adapter_contract",
]
