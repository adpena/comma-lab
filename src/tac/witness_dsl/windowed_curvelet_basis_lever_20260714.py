# SPDX-License-Identifier: MIT
"""DSL basis lever for the selected windowed-directional frame (task #502).

The DSL must HOLD every designed lever (operator 2026-07-06). This module registers the
windowed-curvelet frame (``tac.boundary_math.windowed_curvelet_frame``) as a first-class,
default-OFF DSL basis lever so the next witness arm can A/B it. The nilary
``curriculum_dsl.WindowedCurveletBasis`` wrapper makes registry/activation consumers discover it;
both compile through the same typed ``BasisLeverSpec`` and real trainer flag.

STATE (honest): trainer MLX/NumPy parity and generated-inflate op parity are wired behind the real
``--basis windowed_curvelet`` flag.  The lever is default-OFF because no baseline program composes
it; calling the factory explicitly returns the ACTIVE treatment.  The only remaining
``owed_wire()`` row is the operator-GO n600 byte-closed realized-through-R measurement.  No capacity
probe, receiver advisory, or mask proxy is promoted to a score/family verdict.

means != ends: pointer UNMOVED (0.19108 submittable / 0.18804 bank). The realized d_seg through-R
is OWED (needs a run; operator-GO / CONTAINMENT).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tac.boundary_math.windowed_curvelet_frame import (
    WindowedCurveletConfig,
    localization_certificate,
    n_atoms,
)
from tac.witness_dsl.basis_control import genuine_frame_windowed_curvelet_config
from tac.witness_dsl.curriculum_dsl import Lever, real_trainer_flags
from tac.witness_dsl.optimal_basis_20260714 import BasisFamily, BasisLeverSpec

# Historical capacity/localization constants retained for API compatibility.  They are upper-bound
# spatial-dictionary evidence only, not through-R score authority and not a family-win claim.
LOCALIZATION_ENVELOPE_SPAN = 1.0            # vs polar-Fourier 1.5e-7 (the swap-test margin)
SPECTRAL_CAPACITY_GAIN = (1.7, 2.0)        # reverse-waterfill B_iso/B_orient (n600)
SPATIAL_NTERM_CAPACITY_GAIN_N600 = 1.09    # OMP K_fourier/K_curvelet at rel-err<=0.10 (n600)

WIRE_STATUS_READY = "TRAIN_INFLATE_OP_PARITY_READY_THROUGHR_OWED"


class WindowedCurveletWireNotReady(RuntimeError):
    """Compatibility exception for unsupported custom frame configs."""


@dataclass(frozen=True)
class WindowedCurveletBasisLeverSpec:
    """Typed windowed-curvelet treatment; default-off is program composition, not a fake no-op."""

    config: WindowedCurveletConfig = field(default_factory=genuine_frame_windowed_curvelet_config)
    enabled: bool = True
    wire_status: str = WIRE_STATUS_READY

    def certificate_passes(self) -> bool:
        """Confirm the held config is spatially localized (not a literal-family claim)."""
        return bool(localization_certificate(self.config).passes)

    def compile_lever(self) -> Lever:
        """Compile to the canonical real trainer flag, or an explicit disabled no-op."""
        if self.config != genuine_frame_windowed_curvelet_config():
            raise WindowedCurveletWireNotReady(
                "custom windowed-curvelet configs are not serialized by the current trainer flag; "
                "use the sealed default config or extend train+checkpoint+inflate together"
            )
        canonical = BasisLeverSpec(family=BasisFamily.WINDOWED_CURVELET).compile_lever()
        overrides: dict = dict(canonical.overrides) if self.enabled else {}
        # Guard against any future invented flag sneaking in (never-invent-flags).
        unknown = sorted(set(overrides) - set(real_trainer_flags()))
        if unknown:
            raise RuntimeError(f"windowed-curvelet lever invented trainer flags: {unknown}")
        notes = (
            f"task502 selected windowed-directional frame; certificate_passes="
            f"{self.certificate_passes()}; DEFAULT-OFF by baseline composition; wire_status="
            f"{self.wire_status}; enabled={self.enabled}. Capacity evidence is UPPER-BOUND-only, "
            f"not a through-R row. Byte-closed n600 realized d_seg OWED. "
            f"n_atoms={n_atoms(self.config)}."
        )
        return Lever(name="basis_family::windowed_curvelet", overrides=overrides, notes=notes)

    def owed_wire(self) -> tuple[str, ...]:
        """The exact landings needed to make this a REAL, composable, A/B-able basis arm."""
        return (
            "measure: real n600 byte-closed through-R d_seg A/B (windowed_curvelet vs "
            "legacy_fourier_ab_control) under the governed launcher -- ONLY score-authority row; "
            "operator-GO / CONTAINMENT, PREPARED_NOT_FIRED",
        )


def windowed_curvelet_basis_lever(
    config: WindowedCurveletConfig | None = None,
) -> Lever:
    """Convenience factory for the explicit treatment (baseline omission keeps it default-OFF)."""
    spec = WindowedCurveletBasisLeverSpec(config=config or genuine_frame_windowed_curvelet_config())
    return spec.compile_lever()


__all__ = [
    "LOCALIZATION_ENVELOPE_SPAN",
    "SPATIAL_NTERM_CAPACITY_GAIN_N600",
    "SPECTRAL_CAPACITY_GAIN",
    "WIRE_STATUS_READY",
    "WindowedCurveletBasisLeverSpec",
    "WindowedCurveletWireNotReady",
    "windowed_curvelet_basis_lever",
]
