# SPDX-License-Identifier: MIT
"""X-ray primitive for exact-axis pairset component marginals."""
from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tac.optimization.pairset_component_marginal import (
    PAIRSET_COMPONENT_MARGINAL_MODEL_SCHEMA,
    PAIRSET_COMPONENT_MARGINAL_XRAY_PRIMITIVE_NAME,
    canonical_signal_refs,
)
from tac.xray.base import ComposedXRayPrimitive, WireInHook, XRayPrimitiveResult


@dataclass(frozen=True)
class PairsetComponentMarginalReport:
    """Compact typed report consumed by xray wire-in adapters."""

    active: bool
    axes: tuple[str, ...]
    training_row_count: int
    safe_drop_pair_indices_by_axis: Mapping[str, tuple[int, ...]]
    protected_drop_pair_indices_by_axis: Mapping[str, tuple[int, ...]]
    cross_axis_transfer_statuses: tuple[str, ...]
    canonical_signal_refs: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.training_row_count < 0:
            raise ValueError("training_row_count must be non-negative")


class PairsetComponentMarginalXRay:
    """Canonical xray primitive over ``pairset_component_marginal_model.v1``."""

    @property
    def name(self) -> str:
        return PAIRSET_COMPONENT_MARGINAL_XRAY_PRIMITIVE_NAME

    @property
    def wire_in_hooks(self) -> tuple[WireInHook, ...]:
        return (
            "sensitivity_map",
            "pareto_constraint",
            "bit_allocator",
            "cathedral_autopilot",
            "continual_learning",
            "probe_disambiguator",
        )

    def _load_payload(self, target: Path | str | Mapping[str, Any]) -> tuple[dict[str, Any], Path | None]:
        if isinstance(target, Mapping):
            return dict(target), None
        path = Path(target)
        if not path.is_file():
            raise ValueError(f"pairset component marginal target {path} does not exist")
        return json.loads(path.read_text(encoding="utf-8")), path

    def _extract_model(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        if payload.get("schema") == PAIRSET_COMPONENT_MARGINAL_MODEL_SCHEMA:
            return dict(payload)
        observation_feedback = payload.get("observation_feedback")
        if isinstance(observation_feedback, Mapping):
            model = observation_feedback.get("pairset_component_marginal_model")
            if isinstance(model, Mapping):
                return dict(model)
        raise ValueError(
            "target must be pairset_component_marginal_model.v1 or a portfolio "
            "with observation_feedback.pairset_component_marginal_model"
        )

    def compute(
        self,
        target: Path | str | Mapping[str, Any],
        **_kwargs: Any,
    ) -> XRayPrimitiveResult:
        payload, path = self._load_payload(target)
        model = self._extract_model(payload)
        axis_models = model.get("axis_models")
        if not isinstance(axis_models, Mapping):
            axis_models = {}
        safe_by_axis: dict[str, tuple[int, ...]] = {}
        protected_by_axis: dict[str, tuple[int, ...]] = {}
        for axis, axis_model in sorted(axis_models.items()):
            if not isinstance(axis_model, Mapping):
                continue
            safe_by_axis[str(axis)] = tuple(
                int(value) for value in axis_model.get("safe_drop_pair_indices", ())
            )
            protected_by_axis[str(axis)] = tuple(
                int(value) for value in axis_model.get("protected_drop_pair_indices", ())
            )
        transfer_statuses = []
        for row in model.get("cross_axis_transfer_diagnostics", ()):
            if isinstance(row, Mapping):
                transfer_statuses.append(str(row.get("transfer_status") or "unknown"))
        report = PairsetComponentMarginalReport(
            active=model.get("active") is True,
            axes=tuple(str(axis) for axis in model.get("axes", ())),
            training_row_count=int(model.get("training_row_count") or 0),
            safe_drop_pair_indices_by_axis=safe_by_axis,
            protected_drop_pair_indices_by_axis=protected_by_axis,
            cross_axis_transfer_statuses=tuple(sorted(transfer_statuses)),
            canonical_signal_refs=canonical_signal_refs(),
        )
        return XRayPrimitiveResult(
            primitive_name=self.name,
            archive_or_video_path=path,
            archive_sha256=None,
            primitive_value=report,
            evidence_grade="empirical-anchor" if report.active else "proxy",
            confidence_band=None,
            composes_with=(
                "per_pair_score_decomposition",
                "segnet_margin_polytope",
                "posenet_se3_lie_algebra",
                "score_lipschitz",
            ),
            wire_in_hooks_engaged=self.wire_in_hooks,
            metadata={
                "model_schema": model.get("schema"),
                "allowed_use": model.get("allowed_use"),
                "identity_policy": model.get("identity_policy"),
                "score_claim": False,
                "ready_for_exact_eval_dispatch": False,
            },
        )

    def compose_with(self, other):
        return ComposedXRayPrimitive(self, other)


__all__ = [
    "PairsetComponentMarginalReport",
    "PairsetComponentMarginalXRay",
]
