# SPDX-License-Identifier: MIT
"""DSL basis lever for the GENUINELY LOCALIZED windowed-curvelet frame (task #502).

The DSL must HOLD every designed lever (operator 2026-07-06). This module registers the
windowed-curvelet frame (``tac.boundary_math.windowed_curvelet_frame``) as a first-class,
default-OFF DSL basis lever so the next witness arm can A/B it -- WITHOUT touching the hot
``curriculum_dsl`` module and WITHOUT inventing trainer flags.

STATE (honest): the windowed-curvelet feats are NOT yet implemented in the trainer forward
or the generated ``inflate.py``. Until that op-parity + a real n600 through-R d_seg receipt
exist, this lever is BYTE-IDENTICAL: ``compile_lever()`` returns EMPTY overrides (a no-op that
perturbs no sealed config, changes no archive bytes). It carries the designed
``WindowedCurveletConfig`` and the measured capacity/localization evidence as metadata, plus an
explicit OWED-wire list -- so the design is held, queryable, and never orphaned, but no fake
"active" claim is made (catalog #351: an unlocalized/unwired frame must never carry an active
label). ``owed_wire()`` names the exact landings needed to make it a real, composable A/B arm.

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
from tac.witness_dsl.curriculum_dsl import Lever, real_trainer_flags

# Measured evidence (advisory; not score authority) carried as lever metadata so the
# activation ledger / next arm sees WHY this lever is worth firing.
LOCALIZATION_ENVELOPE_SPAN = 1.0            # vs polar-Fourier 1.5e-7 (the swap-test margin)
SPECTRAL_CAPACITY_GAIN = (1.7, 2.0)        # reverse-waterfill B_iso/B_orient (n600)
SPATIAL_NTERM_CAPACITY_GAIN_N600 = 1.09    # OMP K_fourier/K_curvelet at rel-err<=0.10 (n600)

WIRE_STATUS_OWED = "OWED_TRAIN_INFLATE_OP_PARITY"


class WindowedCurveletWireNotReady(RuntimeError):
    """Raised only if a caller forces a non-byte-identical compile before the wire lands."""


@dataclass(frozen=True)
class WindowedCurveletBasisLeverSpec:
    """Typed, default-OFF windowed-curvelet basis lever.

    Holds the designed ``WindowedCurveletConfig``. ``compile_lever()`` is byte-identical
    (empty overrides) until the trainer + generated inflate implement windowed-curvelet feats
    with op-parity. ``enabled`` defaults False; setting it True without the wire raises (there
    is no honest active compile yet -- the OWED list must be closed first).
    """

    config: WindowedCurveletConfig = field(default_factory=WindowedCurveletConfig)
    enabled: bool = False
    wire_status: str = WIRE_STATUS_OWED

    def certificate_passes(self) -> bool:
        """Confirm the held config produces a GENUINELY localized frame (the anti-fake gate)."""
        return bool(localization_certificate(self.config).passes)

    def compile_lever(self) -> Lever:
        """Return the DSL Lever. Byte-identical (empty overrides) while the wire is OWED.

        A future arm that lands the trainer/inflate flags flips ``wire_status`` and populates
        real overrides here; until then composing this lever changes nothing.
        """
        if self.enabled and self.wire_status == WIRE_STATUS_OWED:
            raise WindowedCurveletWireNotReady(
                "windowed-curvelet feats are not yet wired into the trainer/inflate (op-parity + "
                "through-R receipt OWED); refusing to compile an 'active' lever. Close owed_wire() "
                "first, or keep enabled=False for the byte-identical held lever."
            )
        overrides: dict = {}  # byte-identical: no sealed-config perturbation, no archive-byte change
        # Guard against any future invented flag sneaking in (never-invent-flags).
        unknown = sorted(set(overrides) - set(real_trainer_flags()))
        if unknown:
            raise RuntimeError(f"windowed-curvelet lever invented trainer flags: {unknown}")
        notes = (
            f"task502 windowed-curvelet frame (GENUINELY localized; certificate_passes="
            f"{self.certificate_passes()}); DEFAULT-OFF byte-identical held lever; wire_status="
            f"{self.wire_status}. Measured (advisory): envelope span {LOCALIZATION_ENVELOPE_SPAN} vs "
            f"Fourier 1.5e-7; spectral capacity gain {SPECTRAL_CAPACITY_GAIN[0]}-"
            f"{SPECTRAL_CAPACITY_GAIN[1]}x; spatial OMP N-term ~{SPATIAL_NTERM_CAPACITY_GAIN_N600}x "
            f"(n600). Realized d_seg through-R OWED. n_atoms={n_atoms(self.config)}."
        )
        return Lever(name="basis_family::windowed_curvelet", overrides=overrides, notes=notes)

    def owed_wire(self) -> tuple[str, ...]:
        """The exact landings needed to make this a REAL, composable, A/B-able basis arm."""
        return (
            "trainer: add windowed_curvelet_feats forward + a --windowed-curvelet front-end flag "
            "(register the flag in the DSL real_trainer_flags set -- never invent it inline)",
            "inflate: add the op-parity windowed-curvelet feats regeneration to "
            "tools/levelset_byte_close_and_eval.py (deterministic, rule-118 free bank)",
            "parity: bit-exact numpy<->trainer<->inflate feats check (mlx_parity_check exists for "
            "the primitive; extend it to the trainer/inflate path)",
            "measure: real n600 through-R d_seg A/B (curvelet front-end vs polar-Fourier) at "
            "matched counted bytes -- the ONLY score authority (operator-GO / CONTAINMENT)",
            "sweep: (w0, width_ratio, aniso, n_scales, n_orient0, n_trans, f0) at optimal form "
            "before any adopt/kill verdict (optimal-form discipline)",
        )


def windowed_curvelet_basis_lever(
    config: WindowedCurveletConfig | None = None,
) -> Lever:
    """Convenience factory: the default-OFF byte-identical windowed-curvelet DSL lever."""
    spec = WindowedCurveletBasisLeverSpec(config=config or WindowedCurveletConfig())
    return spec.compile_lever()


__all__ = [
    "LOCALIZATION_ENVELOPE_SPAN",
    "SPATIAL_NTERM_CAPACITY_GAIN_N600",
    "SPECTRAL_CAPACITY_GAIN",
    "WIRE_STATUS_OWED",
    "WindowedCurveletBasisLeverSpec",
    "WindowedCurveletWireNotReady",
    "windowed_curvelet_basis_lever",
]
