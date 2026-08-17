# SPDX-License-Identifier: MIT
"""Exact-arithmetic contest-score model for the ddm_me1 micro-edit engine.

Every number here is DERIVED from the canonical contest constants in
``tac.contest_oracle.constants`` (which mirror the pinned ``upstream/evaluate.py``).
No constant is typed by hand; no number is borrowed from a memo.

``upstream/evaluate.py`` computes::

    S = 100*d_seg + sqrt(10*d_pose) + 25*archive_bytes/37_545_489

The three marginals that govern every micro-edit decision:

* one net Seg flip  -> ``CONTEST_SEG_WEIGHT / CONTEST_PER_ARCHIVE_PIXEL_CELLS``
* one archive byte  -> ``CONTEST_RATE_WEIGHT / CONTEST_RATE_DENOM_BYTES``
* one unit d_pose   -> ``5 / sqrt(10*d_pose)``  (state dependent, DIVERGES as d_pose -> 0)

The seg and rate marginals are base-INDEPENDENT constants. The pose marginal is a
function of the operating point, so it is never cached across bases -- callers pass
the base and get the marginal at that base. This is the cross-regime constant-transfer
guard (memory ``cross-regime-constant-transfer-genus-finishing-stage``): a latched
pose marginal is a bug.

Arithmetic uses ``decimal.Decimal`` at 50 digits throughout. Float is used only where
``math.sqrt`` is unavoidable, and then only via :func:`_dec_sqrt` which computes the
square root inside the Decimal context (exact to working precision, no float round
trip). Rationale: eu4 established that composed micro-edit deltas live at 1e-7..1e-5
against an S of ~0.16, so naive float accumulation loses the sign of the quantity we
are deciding on.

STORES CONSULTED
----------------
* ``upstream/evaluate.py`` (pinned snapshot) -- the score definition, read-only.
* ``tac.contest_oracle.constants`` -- canonical mirrored constants.
* ``.omx/research/ddm_eu4_fresh_eyes_fractal_composition_20260813.md`` -- the exact
  marginal ladder and the union-gating law (composition must be REMEASURED, never
  summed; bank-only >=2e-5 REFUTED).
* ``.omx/research/ddm_rr4_t4_verdict_pointer_move_20260817.md`` -- the live base.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, getcontext, localcontext
from typing import Final

from tac.contest_oracle.constants import (
    CONTEST_PER_ARCHIVE_PIXEL_CELLS,
    CONTEST_POSE_SQRT_INNER,
    CONTEST_RATE_DENOM_BYTES,
    CONTEST_RATE_WEIGHT,
    CONTEST_SEG_WEIGHT,
)

__all__ = [
    "BREAKEVEN_FLIPS_PER_BYTE",
    "CANONICAL_NOISE_BAND_S",
    "NAMING_BAR_S",
    "PRECISION",
    "RATE_PER_BYTE_S",
    "SEG_PER_FLIP_S",
    "ScoreDelta",
    "ScoreState",
    "compose_deltas_unverified",
    "pose_marginal",
    "pose_term",
    "rate_term",
    "seg_term",
]

PRECISION: Final[int] = 50
getcontext().prec = PRECISION

# --- base-independent exact marginals (derived, never typed) ------------------

SEG_PER_FLIP_S: Final[Decimal] = Decimal(int(CONTEST_SEG_WEIGHT)) / Decimal(
    CONTEST_PER_ARCHIVE_PIXEL_CELLS
)
"""S cost of one net SegNet argmax flip. ``100 / 117_964_800`` ~= 8.4771e-7."""

RATE_PER_BYTE_S: Final[Decimal] = Decimal(int(CONTEST_RATE_WEIGHT)) / Decimal(
    CONTEST_RATE_DENOM_BYTES
)
"""S cost of one archive byte. ``25 / 37_545_489`` ~= 6.6586e-7."""

BREAKEVEN_FLIPS_PER_BYTE: Final[Decimal] = RATE_PER_BYTE_S / SEG_PER_FLIP_S
"""Flips that must be bought per byte spent to break even. ~= 0.78548 flips/B.

Equivalently 1.27311 B may be spent per net flip gained. An edit family whose
realized flips-per-byte sits below this ratio LOSES score even when it lowers
d_seg -- the single most common micro-edit accounting error.
"""

NAMING_BAR_S: Final[Decimal] = Decimal("1e-5")
"""The >=1e-5 |Delta S| bar a composed candidate must clear before it is named
and a T4 fire-order is sealed (charter section 6)."""

CANONICAL_NOISE_BAND_S: Final[Decimal] = Decimal("3.5e-6")
"""Canonical +/- component-reconstruction band. A |Delta S| inside this band is
NOT a result; it is banked calibration (the qs2 disposition in eu4 lens 1)."""


def _dec_sqrt(value: Decimal) -> Decimal:
    """Square root inside the Decimal context -- no float round trip."""
    if value < 0:
        raise ValueError(f"sqrt of negative: {value}")
    with localcontext() as ctx:
        ctx.prec = PRECISION + 10
        return +value.sqrt()


def seg_term(d_seg: Decimal) -> Decimal:
    """``100 * d_seg``."""
    return Decimal(int(CONTEST_SEG_WEIGHT)) * d_seg


def pose_term(d_pose: Decimal) -> Decimal:
    """``sqrt(10 * d_pose)``."""
    return _dec_sqrt(Decimal(CONTEST_POSE_SQRT_INNER) * d_pose)


def rate_term(archive_bytes: int) -> Decimal:
    """``25 * archive_bytes / 37_545_489``."""
    return Decimal(int(CONTEST_RATE_WEIGHT)) * Decimal(archive_bytes) / Decimal(
        CONTEST_RATE_DENOM_BYTES
    )


def pose_marginal(d_pose: Decimal) -> Decimal:
    """``dS/d(d_pose) = 5 / sqrt(10*d_pose)`` at the given operating point.

    State dependent BY CONSTRUCTION -- callers must pass the base they are actually
    standing on. Never cache this across bases (cross-regime constant transfer).
    """
    if d_pose <= 0:
        raise ValueError("pose marginal diverges at d_pose <= 0; pass the real base")
    return Decimal(5) / pose_term(d_pose)


@dataclass(frozen=True)
class ScoreState:
    """A complete, self-consistent (d_seg, d_pose, bytes) operating point.

    Constructed from MEASURED components only. ``label`` records the instrument and
    axis so a state can never be silently mixed across instruments (the
    apples-to-apples discipline).
    """

    d_seg: Decimal
    d_pose: Decimal
    archive_bytes: int
    label: str
    archive_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.d_seg < 0 or self.d_pose < 0:
            raise ValueError("distortions must be non-negative")
        if self.archive_bytes <= 0:
            raise ValueError("archive_bytes must be positive")
        if not self.label:
            raise ValueError("label is required: it carries the axis + instrument")

    @property
    def score(self) -> Decimal:
        return seg_term(self.d_seg) + pose_term(self.d_pose) + rate_term(
            self.archive_bytes
        )

    @property
    def pose_marginal(self) -> Decimal:
        return pose_marginal(self.d_pose)

    def gap_to(self, target: Decimal) -> Decimal:
        """Signed distance above ``target`` (positive == still above)."""
        return self.score - target

    def axis_split(self, target: Decimal) -> dict[str, Decimal]:
        """Decompose the gap to ``target`` by axis.

        Returns the three terms, the gap, the pose share OF THE GAP, and the
        residual that survives a hypothetical PERFECT pose -- expressed in net seg
        flips and in bytes. That residual is the honest scope of any micro-edit
        campaign once pose is solved, and it moves whenever the rate term moves,
        so it is recomputed per base rather than quoted from a memo.
        """
        seg = seg_term(self.d_seg)
        pose = pose_term(self.d_pose)
        rate = rate_term(self.archive_bytes)
        gap = self.score - target
        residual = gap - pose
        return {
            "seg_term": seg,
            "pose_term": pose,
            "rate_term": rate,
            "score": seg + pose + rate,
            "gap": gap,
            "pose_share_of_gap": (pose / gap) if gap != 0 else Decimal(0),
            "residual_after_perfect_pose": residual,
            "residual_in_seg_flips": residual / SEG_PER_FLIP_S,
            "residual_in_bytes": residual / RATE_PER_BYTE_S,
        }


@dataclass(frozen=True)
class ScoreDelta:
    """A realized delta measured on ONE object against ONE base.

    ``realized`` distinguishes a MEASURED delta (compile -> decode -> advisory) from a
    projected one. Only realized deltas may be admitted (the pk3/pk4 law).
    """

    d_seg_delta: Decimal
    d_pose_delta: Decimal
    bytes_delta: int
    base: ScoreState
    realized: bool
    provenance: str

    @property
    def net_seg_flips(self) -> Decimal:
        """Signed net flips implied by ``d_seg_delta`` (negative == flips fixed)."""
        return self.d_seg_delta * Decimal(CONTEST_PER_ARCHIVE_PIXEL_CELLS)

    @property
    def delta_s(self) -> Decimal:
        """EXACT Delta S -- recomputes the full score at the perturbed state.

        Deliberately NOT a linearisation: the pose term is a square root, so a
        first-order pose estimate is wrong at exactly the operating point where pose
        dominates. The engine always evaluates the true nonlinear difference.
        """
        after = ScoreState(
            d_seg=self.base.d_seg + self.d_seg_delta,
            d_pose=self.base.d_pose + self.d_pose_delta,
            archive_bytes=self.base.archive_bytes + self.bytes_delta,
            label=self.base.label,
        )
        return after.score - self.base.score

    @property
    def clears_naming_bar(self) -> bool:
        return self.delta_s <= -NAMING_BAR_S

    @property
    def inside_noise_band(self) -> bool:
        return abs(self.delta_s) < CANONICAL_NOISE_BAND_S

    def axis_contributions(self) -> dict[str, Decimal]:
        """Split Delta S into its three axis contributions (they sum to delta_s)."""
        seg_c = seg_term(self.base.d_seg + self.d_seg_delta) - seg_term(self.base.d_seg)
        pose_c = pose_term(self.base.d_pose + self.d_pose_delta) - pose_term(
            self.base.d_pose
        )
        rate_c = rate_term(self.base.archive_bytes + self.bytes_delta) - rate_term(
            self.base.archive_bytes
        )
        return {"seg": seg_c, "pose": pose_c, "rate": rate_c, "total": seg_c + pose_c + rate_c}


def compose_deltas_unverified(
    deltas: list[ScoreDelta], base: ScoreState
) -> ScoreDelta:
    """Sum independent deltas onto a shared base -- a PROJECTION, never a verdict.

    eu4 established the union-gating law: composition must be REMEASURED on the
    joint object, never summed, because (a) the pose term is nonlinear so pose
    contributions do not add, and (b) qs4 measured that a compensation solved for
    one object costs +2.4e-4 when carried to another.

    This helper therefore returns ``realized=False`` unconditionally and stamps the
    provenance as a projection. It exists to ORDER the evaluation queue, and its
    output may never be admitted. Call sites that want a verdict must re-compile and
    re-measure the union.
    """
    if not deltas:
        raise ValueError("no deltas to compose")
    for delta in deltas:
        if delta.base != base:
            raise ValueError(
                "cannot compose deltas measured against different bases; "
                f"expected {base.label!r}, got {delta.base.label!r}"
            )
    return ScoreDelta(
        d_seg_delta=sum((d.d_seg_delta for d in deltas), Decimal(0)),
        d_pose_delta=sum((d.d_pose_delta for d in deltas), Decimal(0)),
        bytes_delta=sum(d.bytes_delta for d in deltas),
        base=base,
        realized=False,
        provenance=(
            "PROJECTION_ONLY additive union of "
            + ",".join(d.provenance for d in deltas)
            + " -- NOT a verdict; the union must be recompiled and remeasured"
        ),
    )
