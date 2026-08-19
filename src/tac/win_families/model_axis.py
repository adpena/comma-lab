# SPDX-License-Identifier: MIT
"""F4 -- the MODEL-AXIS RECODER: probability-model wins, honestly deflated.

THE FAMILY
----------
``ddm_fx1`` and ``ddm_ma1`` are one surface: the payload and the container are fixed, and
the only thing changing is the PROBABILITY MODEL the arithmetic coder is driven by.

``ddm_fx1``  the fixed-point log-odds (logistic) mixer.  It reopened an axis that four
             ``ddm_me1`` architectures had closed, because those architectures could only
             express a weighted ARITHMETIC mean of odds multipliers and a mean can never
             leave the convex hull of its members.  Real mixing adds in the log-odds
             domain -- a weighted GEOMETRIC mean -- which SHARPENS past any member.
             Restricting the weights to a dyadic grid turns that into repeated ``sqrt``,
             which IEEE-754 requires to be correctly rounded, so the mixer is
             bit-identical across platforms without ``log``/``exp``.
             MEASURED: **-560 B** code length, **+127.3 s** decode against 297.7 s of
             local headroom.
``ddm_ma1``  the within-miss relative law: the same Krichevsky-Trofimov count ratio the
             archive already uses for the hit event, pointed at the miss sector nobody
             had pointed it at.  MEASURED: **-100.4 B**, ceiling 1,247 B.

THE TWO ACCOUNTING RULES THIS MODULE ENFORCES
---------------------------------------------
**1. A modelled code-length win is not an archive win -- it DEFLATES.**  ``ddm_fx1``
measured a parse-back calibration of **x1.260** between the modelled code length and what
the real container carries.  A reservoir priced at the perfect-model ceiling and quoted
undeflated is the same over-claim as pricing a payload delta as an archive delta (F3's
law).  :class:`ModelAxisReservoir` therefore refuses to project an archive number without
an explicit :class:`Calibration`.

**2. The calibration is an INPUT WITH PROVENANCE, never an inherited constant.**  This is
``ddm_ma1``'s own scar: at ``mc=32`` an inherited constant produced a **FALSE SATURATION
VERDICT** -- the arm concluded the sector was exhausted when the constant, not the sector,
was the binding thing.  That is the cross-regime constant-transfer genus.  So
:class:`Calibration` requires a non-empty ``source`` and the regime it was measured in,
and :meth:`Calibration.assert_applicable` refuses to apply a calibration measured in one
regime to another without an explicit acknowledgement.

WHAT THIS MODULE DOES NOT DO
----------------------------
It does not implement a coder, a mixer, or a context model -- ``ddm_fx1`` and ``ddm_ma1``
own those, and their math is family-specific (UNIQUE-AND-COMPLETE-PER-METHOD).  This is
the ACCOUNTING surface they share and that both got burned by.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

__all__ = [
    "Calibration",
    "ModelAxisError",
    "ModelAxisReservoir",
    "SectorPrice",
    "bits_to_bytes_ceiling",
]

#: ``upstream/evaluate.py:64`` -- the rate denominator.
CONTEST_UNCOMPRESSED_BYTES = 37_545_489
RATE_WEIGHT = 25.0

#: ``ddm_fx1``'s measured parse-back calibration between modelled code length and the
#: realized container.  Recorded here as a NAMED ANCHOR for reuse, never as a default:
#: constructing a :class:`Calibration` requires stating the source explicitly.
FX1_PARSE_BACK_CALIBRATION = 1.260
FX1_CALIBRATION_REGIME = "rr4_transport_n600_token_stream"


class ModelAxisError(RuntimeError):
    """A model-axis accounting precondition failed.  Always fail closed."""


def bits_to_bytes_ceiling(bits: float) -> int:
    """Whole bytes a bit count occupies, rounded away from zero.

    A coder cannot emit a fraction of a byte, so a modelled bit saving of 7 bits is worth
    at most one byte and possibly zero.
    """
    whole = math.ceil(abs(float(bits)) / 8.0)
    return whole if bits >= 0 else -whole


@dataclass(frozen=True)
class Calibration:
    """The measured factor between a modelled code length and the realized archive.

    Args:
        factor: multiply a MODELLED byte count by this to project the realized one.
            ``ddm_fx1`` measured 1.260 on the rr4 transport.
        source: the artifact that measured it.  Required and non-empty -- an
            uncredited factor is an inherited constant, which is the ``ddm_ma1``
            false-saturation defect.
        regime: what body/stream/context it was measured on.  Required for the same
            reason: a factor is a property of a regime, not of the universe.
    """

    factor: float
    source: str
    regime: str

    def __post_init__(self) -> None:
        if not math.isfinite(self.factor) or self.factor <= 0.0:
            raise ModelAxisError(f"calibration factor must be finite and > 0, got {self.factor}")
        if not self.source.strip():
            raise ModelAxisError(
                "a calibration needs a source artifact. An uncredited factor is an "
                "inherited constant, and at mc=32 an inherited constant produced "
                "ddm_ma1's FALSE SATURATION VERDICT."
            )
        if not self.regime.strip():
            raise ModelAxisError(
                "a calibration needs the regime it was measured in. A factor is a "
                "property of a regime; carrying it across regimes is the "
                "cross-regime constant-transfer genus."
            )

    def assert_applicable(self, regime: str, *, allow_cross_regime: bool = False) -> None:
        """Refuse to apply this calibration outside its measured regime.

        Args:
            regime: the regime the caller is about to apply it in.
            allow_cross_regime: explicit acknowledgement that the transfer is intended
                and unproven.  Naming it is the point.

        Raises:
            ModelAxisError: regimes differ and the transfer was not acknowledged.
        """
        if regime == self.regime or allow_cross_regime:
            return
        raise ModelAxisError(
            f"calibration {self.factor} was measured on regime {self.regime!r} "
            f"(source {self.source}) and is being applied to {regime!r}. "
            "Re-measure it, or pass allow_cross_regime=True to record that the transfer "
            "is intended and unproven."
        )

    def to_json(self) -> dict[str, Any]:
        return {"factor": self.factor, "source": self.source, "regime": self.regime}


@dataclass(frozen=True)
class SectorPrice:
    """One sector of the coded stream, with its ceiling and what has been realized.

    ``ceiling_bytes`` is the perfect-model bound -- what the sector could yield if the
    model were exact.  ``realized_bytes`` is what a built mechanism actually measured.
    The gap between them is headroom, and quoting the ceiling as if it were the win is the
    over-claim this type exists to make awkward.
    """

    name: str
    ceiling_bytes: float
    realized_bytes: float = 0.0

    def __post_init__(self) -> None:
        if self.ceiling_bytes < 0:
            raise ModelAxisError(f"sector {self.name!r} has a negative ceiling")
        if self.realized_bytes < 0:
            raise ModelAxisError(
                f"sector {self.name!r} has negative realized bytes; realized savings are "
                "recorded as positive magnitudes"
            )
        if self.realized_bytes > self.ceiling_bytes:
            raise ModelAxisError(
                f"sector {self.name!r} realized {self.realized_bytes} B against a ceiling "
                f"of {self.ceiling_bytes} B. Either the ceiling is wrong or the realized "
                "number is not measuring the same sector."
            )

    @property
    def headroom_bytes(self) -> float:
        return self.ceiling_bytes - self.realized_bytes

    @property
    def realized_fraction(self) -> float:
        """Share of the ceiling actually taken.  Zero-ceiling sectors report 0.0."""
        return self.realized_bytes / self.ceiling_bytes if self.ceiling_bytes else 0.0

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ceiling_bytes": self.ceiling_bytes,
            "realized_bytes": self.realized_bytes,
            "headroom_bytes": self.headroom_bytes,
            "realized_fraction": self.realized_fraction,
        }


class ModelAxisReservoir:
    """The deflated-reservoir account for a set of model-axis sectors."""

    def __init__(self, sectors: list[SectorPrice] | tuple[SectorPrice, ...]) -> None:
        if not sectors:
            raise ModelAxisError("a reservoir needs at least one sector")
        names = [sector.name for sector in sectors]
        if len(set(names)) != len(names):
            raise ModelAxisError(f"duplicate sector names: {names}")
        self.sectors = tuple(sectors)

    @property
    def ceiling_bytes(self) -> float:
        return sum(sector.ceiling_bytes for sector in self.sectors)

    @property
    def realized_bytes(self) -> float:
        return sum(sector.realized_bytes for sector in self.sectors)

    @property
    def headroom_bytes(self) -> float:
        return self.ceiling_bytes - self.realized_bytes

    def project_archive_bytes(
        self,
        modelled_bytes: float,
        calibration: Calibration,
        *,
        regime: str,
        allow_cross_regime: bool = False,
    ) -> float:
        """Deflate a MODELLED byte count into a projected archive byte count.

        A projection is still a projection: it is not an archive measurement and must
        never be reported as one.  ``ddm_fx1``'s own memo projects the archive as
        ``ceil(code_bytes)`` and keeps the calibration separate for exactly this reason.
        """
        calibration.assert_applicable(regime, allow_cross_regime=allow_cross_regime)
        return float(modelled_bytes) / calibration.factor

    def projected_score_delta(
        self,
        modelled_bytes: float,
        calibration: Calibration,
        *,
        regime: str,
        allow_cross_regime: bool = False,
    ) -> float:
        """Projected rate-leg score delta for a SAVING of ``modelled_bytes``.

        Returns a NEGATIVE number for a saving, matching the score convention.
        """
        projected = self.project_archive_bytes(
            modelled_bytes,
            calibration,
            regime=regime,
            allow_cross_regime=allow_cross_regime,
        )
        return -RATE_WEIGHT * projected / CONTEST_UNCOMPRESSED_BYTES

    def saturated_sectors(self, *, threshold: float = 0.99) -> tuple[str, ...]:
        """Sectors at or above ``threshold`` of their ceiling.

        A saturation verdict is only as good as the ceiling it is measured against.
        ``ddm_ma1`` recorded a FALSE saturation at ``mc=32`` because an inherited constant
        -- not the sector -- was binding, so a caller reading this should ask what set the
        ceiling before concluding a sector is exhausted.
        """
        return tuple(
            sector.name
            for sector in self.sectors
            if sector.realized_fraction >= threshold
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "sectors": [sector.to_json() for sector in self.sectors],
            "ceiling_bytes": self.ceiling_bytes,
            "realized_bytes": self.realized_bytes,
            "headroom_bytes": self.headroom_bytes,
            "axis": "[macOS-CPU exact byte / code-length measurement]",
            "score_claim": False,
            "note": (
                "ceiling is the perfect-model bound, not a win. A modelled code-length "
                "saving deflates into the archive (ddm_fx1 measured x1.260); a saturation "
                "verdict is only as good as the ceiling that set it (ddm_ma1's mc=32 "
                "false saturation came from an inherited constant)."
            ),
        }
