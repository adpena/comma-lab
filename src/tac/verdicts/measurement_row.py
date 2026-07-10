# SPDX-License-Identifier: MIT
"""tac.verdicts.measurement_row — the canonical typed MEASUREMENT row.

A :class:`MeasurementRow` is one measured/derived scalar, carried with EVERYTHING
that makes it honest and load-bearing (the operating-manual §5 discipline: label
every claim by how it was obtained). The fields encode the standing disciplines so
they cannot be silently dropped:

  * ``axis_tag``      — which evidence axis (contest-CPU/CUDA are authority; the
                        macOS / MLX / mask-level / through-R tags are NOT). A bare
                        number with no axis is the metric-laundering bug class.
  * ``provenance``    — git_sha + seed + config_ref + inputs_sha256 + tool, so any
                        host can re-derive the number (deterministic-repro spine).
  * ``noise_floor``   — the composed instrument floor for THIS Δ. Nullable == the
    + ``floor_provenance`` floor is UNKNOWN (honest), but NEVER silently 0: a
                        non-None floor MUST carry its provenance (design philosophy
                        P2 — "every comparison carries its noise floor"; a Δ below
                        an unknown/unstated floor is INSTANCE-level at best).
  * ``n_samples``     — 600 is the n600 authority scale; a subset MUST state its
    + ``n_samples_reason`` reason (CLAUDE.md "allergic to non-n600-scale/toys").
  * ``review_status`` — reviewed / unreviewed_recovery_written / provisional
                        (MEMORY L81: a recovery-written verdict is UNREVIEWED by
                        construction and must SAY SO, and auto-queue fresh eyes).

Frozen + validated in ``__post_init__`` (a malformed row cannot be constructed);
``to_json_dict()`` returns a stable-ordered plain dict for the emission surface.

This module is LEAF-CLEAN: it imports only the stdlib. It does NOT import
``tac.through_r`` or ``tac.session_bus`` (sibling units building in parallel; the
consumption wire-ins land in #389).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

__all__ = [
    "AxisTag",
    "MeasurementRow",
    "MeasurementRowError",
    "Provenance",
    "ReviewStatus",
]

# n600 is the authority scale (all 600 pairs, real gt_n600). A row at any other
# sample count MUST carry a reason (subset probes are a MEANS, never evidence).
N600_EXPECTED = 600

# Length of a hex SHA-256 digest (inputs_sha256 sanity check).
_SHA256_HEXLEN = 64


class MeasurementRowError(ValueError):
    """Raised when a :class:`MeasurementRow` (or :class:`Provenance`) is malformed.

    Subclasses ``ValueError`` so callers already catching ``ValueError`` on typed
    construction keep working.
    """


class AxisTag(StrEnum):
    """The evidence axis a measurement lives on.

    Authority (a real score / promotion / kill may cite it): ``CONTEST_CPU`` and
    ``CONTEST_CUDA`` ONLY — and only on 1:1 contest-compliant hardware. Every other
    tag is a NON-authority research/advisory signal (gradient + prior only), per
    the CLAUDE.md authority ladder + "MPS is NEVER a score authority".
    """

    CONTEST_CPU = "[contest-CPU]"
    CONTEST_CUDA = "[contest-CUDA]"
    MACOS_CPU_ADVISORY = "[macOS-CPU advisory]"
    MACOS_MLX_RESEARCH_SIGNAL = "[macOS-MLX research-signal]"
    MASK_LEVEL = "[mask-level]"
    THROUGH_R = "[through-R]"
    PROXY = "[proxy]"
    PREDICTION = "[prediction]"
    ADVISORY_ONLY = "[advisory only]"

    @property
    def is_authority(self) -> bool:
        """True iff a score/frontier/promotion/kill claim may cite this axis."""
        return self in (AxisTag.CONTEST_CPU, AxisTag.CONTEST_CUDA)

    @classmethod
    def coerce(cls, value: AxisTag | str) -> AxisTag:
        """Accept an :class:`AxisTag` or its exact bracketed string; else raise."""
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            for member in cls:
                if member.value == value:
                    return member
        raise MeasurementRowError(
            f"axis_tag must be an AxisTag or one of "
            f"{[m.value for m in cls]!r}; got {value!r}"
        )


class ReviewStatus(StrEnum):
    """Whether the row has passed fresh-eyes review (MEMORY L81).

    ``REVIEWED``                     — a fresh reviewer verified it.
    ``UNREVIEWED_RECOVERY_WRITTEN``  — written during crash-recovery / by the same
                                       agent that produced it → UNREVIEWED by
                                       construction; MUST auto-queue a review.
    ``PROVISIONAL``                  — depends on an INFERRED/ASSUMED input (the
                                       verdict-clearance ladder) → not load-bearing
                                       until verified.
    """

    REVIEWED = "reviewed"
    UNREVIEWED_RECOVERY_WRITTEN = "unreviewed_recovery_written"
    PROVISIONAL = "provisional"

    @property
    def is_load_bearing(self) -> bool:
        """True iff a decision may rest on this row without further review."""
        return self is ReviewStatus.REVIEWED

    @classmethod
    def coerce(cls, value: ReviewStatus | str) -> ReviewStatus:
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            for member in cls:
                if member.value == value:
                    return member
        raise MeasurementRowError(
            f"review_status must be a ReviewStatus or one of "
            f"{[m.value for m in cls]!r}; got {value!r}"
        )


@dataclass(frozen=True)
class Provenance:
    """Everything a fresh host needs to re-derive the number (deterministic-repro).

    ``git_sha`` and ``tool`` are REQUIRED (the number came from SOME code at SOME
    commit). ``seed`` / ``config_ref`` / ``inputs_sha256`` are nullable but, when
    given, are type/shape-validated so a placeholder cannot masquerade as real
    provenance.
    """

    git_sha: str
    tool: str
    seed: int | None = None
    config_ref: str | None = None
    inputs_sha256: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.git_sha, str) or not self.git_sha.strip():
            raise MeasurementRowError("Provenance.git_sha must be a non-empty string")
        if not isinstance(self.tool, str) or not self.tool.strip():
            raise MeasurementRowError("Provenance.tool must be a non-empty string")
        if self.seed is not None and (isinstance(self.seed, bool) or not isinstance(self.seed, int)):
            raise MeasurementRowError(
                f"Provenance.seed must be an int or None; got {self.seed!r}"
            )
        if self.config_ref is not None and not isinstance(self.config_ref, str):
            raise MeasurementRowError("Provenance.config_ref must be a string or None")
        if self.inputs_sha256 is not None:
            s = self.inputs_sha256
            if not isinstance(s, str) or len(s) != _SHA256_HEXLEN or any(
                c not in "0123456789abcdef" for c in s.lower()
            ):
                raise MeasurementRowError(
                    "Provenance.inputs_sha256 must be a 64-char hex digest or None; "
                    f"got {self.inputs_sha256!r}"
                )

    def to_json_dict(self) -> dict:
        """Stable-ordered plain dict."""
        return {
            "git_sha": self.git_sha,
            "tool": self.tool,
            "seed": self.seed,
            "config_ref": self.config_ref,
            "inputs_sha256": self.inputs_sha256,
        }


@dataclass(frozen=True)
class MeasurementRow:
    """One measured/derived scalar, carried with its full honesty envelope.

    Construct it and it is VALID (``__post_init__`` refuses a malformed row); pass
    it to :func:`tac.verdicts.emit.emit_verdict`. The invariants encode the
    standing disciplines — see the module docstring.
    """

    value: float
    units: str
    axis_tag: AxisTag
    provenance: Provenance
    n_samples: int
    review_status: ReviewStatus
    # P2: nullable == UNKNOWN (honest); a non-None floor MUST carry provenance so
    # a silent 0.0 floor is impossible to construct.
    noise_floor: float | None = None
    floor_provenance: str | None = None
    # A subset (n_samples != 600) MUST state why it is not n600-scale.
    n_samples_reason: str | None = None
    # Free-form label (e.g. "d_seg", "d_pose", "rate", "archive_bytes"). Optional
    # but recommended so a reader knows WHICH quantity the value is.
    quantity: str | None = None

    def __post_init__(self) -> None:
        # value: a finite real number (reject bool, NaN, inf).
        if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
            raise MeasurementRowError(
                f"MeasurementRow.value must be a real number; got {self.value!r}"
            )
        if not math.isfinite(self.value):
            raise MeasurementRowError(
                f"MeasurementRow.value must be finite (no NaN/inf); got {self.value!r}"
            )
        # units: non-empty string.
        if not isinstance(self.units, str) or not self.units.strip():
            raise MeasurementRowError("MeasurementRow.units must be a non-empty string")
        # axis_tag / review_status: coerce so a raw string is accepted but stored
        # as the enum (object.__setattr__ because frozen).
        object.__setattr__(self, "axis_tag", AxisTag.coerce(self.axis_tag))
        object.__setattr__(self, "review_status", ReviewStatus.coerce(self.review_status))
        # provenance: must be a Provenance (typed, already self-validated).
        if not isinstance(self.provenance, Provenance):
            raise MeasurementRowError(
                "MeasurementRow.provenance must be a tac.verdicts.Provenance; "
                f"got {type(self.provenance).__name__}"
            )
        # n_samples: positive int; a subset carries a reason.
        if isinstance(self.n_samples, bool) or not isinstance(self.n_samples, int):
            raise MeasurementRowError(
                f"MeasurementRow.n_samples must be an int; got {self.n_samples!r}"
            )
        if self.n_samples <= 0:
            raise MeasurementRowError(
                f"MeasurementRow.n_samples must be > 0; got {self.n_samples}"
            )
        if self.n_samples != N600_EXPECTED and not (
            isinstance(self.n_samples_reason, str) and self.n_samples_reason.strip()
        ):
            raise MeasurementRowError(
                f"MeasurementRow.n_samples={self.n_samples} != {N600_EXPECTED} "
                "(the n600 authority scale) REQUIRES a non-empty n_samples_reason "
                "(a subset is a MEANS, never evidence — CLAUDE.md 'allergic to "
                "non-n600-scale/toys')"
            )
        # P2 noise floor: nullable == UNKNOWN, but a non-None floor MUST carry
        # provenance — this is what makes a silent 0.0 floor impossible.
        if self.noise_floor is not None:
            if isinstance(self.noise_floor, bool) or not isinstance(self.noise_floor, (int, float)):
                raise MeasurementRowError(
                    f"MeasurementRow.noise_floor must be a real number or None; "
                    f"got {self.noise_floor!r}"
                )
            if not math.isfinite(self.noise_floor) or self.noise_floor < 0:
                raise MeasurementRowError(
                    "MeasurementRow.noise_floor must be a finite value >= 0 (or "
                    f"None for UNKNOWN); got {self.noise_floor!r}"
                )
            if not (isinstance(self.floor_provenance, str) and self.floor_provenance.strip()):
                raise MeasurementRowError(
                    f"MeasurementRow.noise_floor={self.noise_floor} is set but "
                    "floor_provenance is empty — a noise floor (ESPECIALLY 0.0) "
                    "MUST state how it was measured/derived (design philosophy P2: "
                    "never a silent-zero floor). Set noise_floor=None to declare "
                    "the floor UNKNOWN honestly instead."
                )
        elif self.floor_provenance is not None and not isinstance(self.floor_provenance, str):
            raise MeasurementRowError(
                "MeasurementRow.floor_provenance must be a string or None"
            )
        if self.quantity is not None and not isinstance(self.quantity, str):
            raise MeasurementRowError("MeasurementRow.quantity must be a string or None")

    @property
    def floor_is_known(self) -> bool:
        """True iff a noise floor was stated (vs None == UNKNOWN)."""
        return self.noise_floor is not None

    @property
    def is_n600(self) -> bool:
        return self.n_samples == N600_EXPECTED

    def to_json_dict(self) -> dict:
        """Stable-ordered plain dict (JSON-serializable; enums rendered as values)."""
        return {
            "quantity": self.quantity,
            "value": self.value,
            "units": self.units,
            "axis_tag": self.axis_tag.value,
            "is_authority_axis": self.axis_tag.is_authority,
            "provenance": self.provenance.to_json_dict(),
            "noise_floor": self.noise_floor,
            "floor_provenance": self.floor_provenance,
            "floor_is_known": self.floor_is_known,
            "n_samples": self.n_samples,
            "n_samples_reason": self.n_samples_reason,
            "is_n600": self.is_n600,
            "review_status": self.review_status.value,
            "is_load_bearing": self.review_status.is_load_bearing,
        }
