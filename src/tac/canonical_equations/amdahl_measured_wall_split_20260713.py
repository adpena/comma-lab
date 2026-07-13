# SPDX-License-Identifier: MIT
"""Measured-only Amdahl composition for wall time and CPU verdict latency.

The equation deliberately composes *savings on disjoint measured components*,
not a product of isolated speedup guesses.  Two levers that touch the same
component (tile-halo and mixed precision both touch scorer forward, for
example) must be benchmarked jointly and supplied as one measured lever.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

EQUATION_ID = "amdahl_measured_disjoint_wall_split_with_async_cpu_verdict_v2"


@dataclass(frozen=True)
class MeasuredSeconds:
    value: float
    source_artifact: str
    status: str = "MEASURED"

    def __post_init__(self) -> None:
        if self.status != "MEASURED":
            raise ValueError("only status=MEASURED is accepted")
        if not isinstance(self.value, (int, float)) or isinstance(self.value, bool):
            raise TypeError("value must be numeric")
        if not math.isfinite(float(self.value)) or float(self.value) < 0:
            raise ValueError("value must be finite and non-negative")
        if not isinstance(self.source_artifact, str) or not self.source_artifact.strip():
            raise ValueError("source_artifact is required for measured inputs")


@dataclass(frozen=True)
class MeasuredLever:
    name: str
    component_id: str
    wall_seconds: MeasuredSeconds
    accelerated_seconds: MeasuredSeconds

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("lever name must be non-empty")
        if not self.component_id.strip():
            raise ValueError("component_id must be non-empty")
        if self.accelerated_seconds.value > self.wall_seconds.value:
            raise ValueError("accelerated_seconds cannot exceed measured wall_seconds")


@dataclass(frozen=True)
class AmdahlWallSplit:
    baseline_seconds: MeasuredSeconds
    levers: tuple[MeasuredLever, ...] = ()
    async_cpu_verdict_service_seconds: MeasuredSeconds | None = None
    async_cpu_verdict_critical_path_seconds: MeasuredSeconds | None = None

    def __post_init__(self) -> None:
        if self.baseline_seconds.value <= 0:
            raise ValueError("baseline_seconds must be > 0")
        names = [lever.name for lever in self.levers]
        if len(names) != len(set(names)):
            raise ValueError("lever names must be unique")
        components = [lever.component_id for lever in self.levers]
        if len(components) != len(set(components)):
            raise ValueError(
                "lever components must be disjoint; benchmark overlapping levers jointly"
            )
        claimed = sum(float(lever.wall_seconds.value) for lever in self.levers)
        if claimed > float(self.baseline_seconds.value) + 1e-12:
            raise ValueError("measured component seconds cannot exceed baseline")
        if (
            self.async_cpu_verdict_critical_path_seconds
            and self.async_cpu_verdict_critical_path_seconds.value
            > self.baseline_seconds.value
        ):
            raise ValueError("async CPU critical-path seconds cannot exceed baseline")

    def compose(self) -> dict[str, object]:
        """Return an additive, measured-only disjoint-component composition."""
        baseline = float(self.baseline_seconds.value)
        saved = 0.0
        factors: dict[str, dict[str, float | str]] = {}
        for lever in self.levers:
            before = float(lever.wall_seconds.value)
            after = float(lever.accelerated_seconds.value)
            saved += before - after
            factors[lever.name] = {
                "component_id": lever.component_id,
                "component_share": before / baseline,
                "component_speedup": before / after if after else math.inf,
                "saved_seconds": before - after,
            }
        remaining = baseline - saved
        if remaining <= 0:
            raise ValueError("composition produced non-positive wall time")
        service = self.async_cpu_verdict_service_seconds
        critical = self.async_cpu_verdict_critical_path_seconds
        return {
            "equation_id": EQUATION_ID,
            "baseline_seconds": baseline,
            "composed_seconds": remaining,
            "composed_speedup": baseline / remaining,
            "lever_factors": factors,
            "async_cpu_verdict_service_seconds": service.value if service else None,
            "async_cpu_verdict_service_to_training_wall_ratio": (
                service.value / baseline if service else None
            ),
            "async_cpu_verdict_critical_path_seconds": critical.value if critical else None,
            "async_cpu_verdict_critical_path_share": (
                critical.value / baseline if critical else None
            ),
            "inputs_status": "MEASURED_ONLY_DISJOINT_COMPONENTS",
            "overlap_rule": (
                "overlapping levers require one joint measured before/after component"
            ),
        }


__all__ = ["EQUATION_ID", "AmdahlWallSplit", "MeasuredLever", "MeasuredSeconds"]
