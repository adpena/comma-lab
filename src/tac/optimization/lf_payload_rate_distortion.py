"""Evaluator-conditioned reverse-waterfill PLANNER over an SNeRV LF/source-state payload.

THE LAW (the single admission predicate this module computes per candidate action):

    keep payload component c  iff  -ΔS_distortion(c) > 25·Δbytes(c) / 37,545,489
    where ΔS_distortion = 100·Δd_seg + Δsqrt(10·d_pose)

This is SCORER-RESPONSE waterfilling, NOT pixel-variance. A payload section pays
rent only when the distortion it buys (in EXACT contest score units) exceeds the
rate it costs. The estimate of ΔS_distortion per section comes from the MEASURED
scorer spectral atlas (``scorer_spectral_sensitivity.v2``): H_seg / H_pose per
band × orientation × amplitude × channel × frame-incidence cell. The bytes come
from the G1b export-binding section decomposition (``snerv_g1b_export_binding_verdict.v1``).

WHERE THIS SITS (authority-disciplined, per docs/vehicle_operating_system.md):

  * This is the PROPOSAL / PLANNING surface — it runs BEFORE the exact receiver
    re-measure. Every row it emits is a PREDICTION (``false-authority``;
    ``promotable=False``; ``requires_exact_remeasure=True``). It NEVER claims a
    score; it ranks what to DROP / QUANTIZE / RECODE so the downstream exact pass
    measures the smallest, highest-value candidate set first.
  * The DOWNSTREAM authority surface is the canonical waterfill law
    ``tac.optimization.evaluator_action_waterfill.CandidateActionEvaluation``,
    which judges a candidate by EXACT measured d_seg/d_pose/bytes against a base
    archive. This planner's PROPOSAL rows are designed to be re-measured into
    those exact rows once an action is actually applied + re-scored.

FAIL-CLOSED SCOPE RULE (the measured-constant scope discipline, Catalog #385 sister):
an atlas sensitivity is only valid INSIDE its ``measurement_scope`` (the grid the
atlas swept: bands, channel-bases, orientations, frame-incidences, amplitudes,
authority tier). If a payload section's coefficient group falls OUTSIDE the atlas
envelope — a band index the atlas never measured, an amplitude beyond the swept
range, an authority tier mismatch — the estimate would EXTRAPOLATE. We refuse to
fabricate: the action is marked ``atlas_scope_valid=False`` / ``scope_invalid``
and its distortion estimate is recorded as ``None`` (NOT zero, NOT a guess). A
scope-invalid action is never ranked above a scope-valid one; the only honest
downstream move on a scope-invalid section is an exact re-measure.

Authority: ``[macOS-CPU advisory]`` / planning-control false authority. The atlas
itself is ``exact_cpu_advisory`` / ``mechanism_update_eligible`` ONLY; nothing
here updates the score roadmap (per the metric-laundering firewall).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

# Reuse the canonical contest constants + score so this planner and the downstream
# exact waterfiller speak ONE currency (never re-derive the formula).
from tac.optimization.evaluator_action_waterfill import (
    CONTEST_ARCHIVE_RATE_DENOM,
)

_SEG_COEF = 100.0
_POSE_INNER = 10.0
_RATE_COEF = 25.0
CONTEST_BYTE_PRICE = _RATE_COEF / float(CONTEST_ARCHIVE_RATE_DENOM)

# Candidate action kinds this planner proposes over a payload section.
ACTION_DROP = "drop"
"""Remove the section's bytes entirely (Δbytes = -section.bytes). Distortion cost
= the full sensitivity the section's coefficient group buys."""

ACTION_QUANTIZE = "quantize_delta"
"""Coarsen the section by a quantization step Δ; frees a fraction of its bytes and
gives up a fraction of its distortion value (the partial-keep waterfill knob)."""

ACTION_RECODE = "recode"
"""Re-encode the section with a cheaper entropy coder; frees bytes at (ideally)
~zero distortion cost (lossless recode) — the free-bytes branch."""

ACTION_QUANTIZE_CONE_MASKED = "quantize_delta_cone_masked"
"""Coarsen the section's coefficients by a quantization step Δ ONLY at frame1
pixels whose JOINT SAFE CONE radius (``tac.optimization.frame1_joint_safe_cone``)
is >= a threshold (the spatially-free set); preserve full precision on the fragile
set (radius < threshold). This is the SPATIAL refinement of :data:`ACTION_QUANTIZE`:
the band×orientation atlas says *which section* is sensitive, but the cone says
*which frame1 pixel inside that section* has free budget. The Δbytes accounting
charges the section bytes freed (proportional to the free fraction × the coarsen
ratio) MINUS the per-pixel keep/coarsen mask's OWN coding cost (bit-packed +
brotli q=11) — the mask must pay rent: a mask whose coding cost exceeds the bytes
it frees is rejected by THE LAW. The distortion given up is cone-radius-weighted:
only the free-set coefficients are coarsened, so the section's atlas distortion
value is scaled by the FREE fraction (fragile pixels contribute ~0 to the
given-up distortion because their coefficients are preserved)."""

_ACTION_KINDS = (
    ACTION_DROP,
    ACTION_QUANTIZE,
    ACTION_RECODE,
    ACTION_QUANTIZE_CONE_MASKED,
)


class LfPayloadRateDistortionError(ValueError):
    """Raised on malformed planner inputs (fail-closed; never silently coerce)."""


# ---------------------------------------------------------------------------
# Typed inputs.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CoefficientGroup:
    """The spectral identity of a payload section, in atlas-grid coordinates.

    A payload section (e.g. the LF wavelet blob) carries energy concentrated in a
    region of the scorer's spectral grid. To estimate how much distortion that
    section buys, we must place it in the SAME coordinates the atlas measured:
    band index, channel basis + channel, orientation, frame-incidence, and the
    perturbation amplitude (LSB). A section that spans multiple cells declares its
    dominant cell here; the planner sums matched cells when ``band_indices`` lists
    more than one. Coordinates the atlas never swept => scope-invalid (fail-closed).
    """

    band_indices: tuple[int, ...]
    """Atlas band index/indices the section's energy occupies (log-spaced bands)."""

    channel_basis: str = ""
    """``rgb`` / ``yuv`` (atlas grid axis). Empty matches any measured basis."""

    channel: str = ""
    """``all`` / ``y`` etc. (atlas grid axis). Empty matches any measured channel."""

    orientation: str = ""
    """``isotropic`` / ``vertical`` etc. Empty matches any measured orientation."""

    frame_incidence: str = ""
    """``frame1_only`` / ``both_opposite``. Empty matches any measured incidence."""

    amplitude_lsb: float | None = None
    """The section's effective perturbation amplitude (LSB). ``None`` matches any
    measured amplitude; a value OUTSIDE the atlas's swept amplitudes => scope-invalid."""

    def __post_init__(self) -> None:
        if not self.band_indices:
            raise LfPayloadRateDistortionError(
                "CoefficientGroup.band_indices must be non-empty"
            )
        for b in self.band_indices:
            if int(b) < 0:
                raise LfPayloadRateDistortionError(
                    f"band index must be non-negative; got {b!r}"
                )
        if self.amplitude_lsb is not None and float(self.amplitude_lsb) <= 0.0:
            raise LfPayloadRateDistortionError(
                f"amplitude_lsb must be positive when set; got {self.amplitude_lsb!r}"
            )


@dataclass(frozen=True)
class PayloadSection:
    """One byte-accountable section of the SNeRV archive payload.

    ``bytes`` is the measured section size from the G1b export-binding decomposition.
    ``coefficient_group`` places the section in atlas-grid coordinates so its
    distortion value can be estimated (or fail-closed when out of scope).
    ``recodeable_floor_bytes`` (optional) is the smallest the section could become
    under a lossless recode (used by the RECODE action); ``None`` means unknown.
    """

    name: str
    bytes: int
    coefficient_group: CoefficientGroup
    recodeable_floor_bytes: int | None = None
    droppable: bool = True
    """Some sections (header / metadata / contest-required) cannot be dropped."""

    def __post_init__(self) -> None:
        if not str(self.name):
            raise LfPayloadRateDistortionError("PayloadSection.name must be non-empty")
        if int(self.bytes) < 0:
            raise LfPayloadRateDistortionError(
                f"PayloadSection.bytes must be non-negative; got {self.bytes!r}"
            )
        if (
            self.recodeable_floor_bytes is not None
            and int(self.recodeable_floor_bytes) < 0
        ):
            raise LfPayloadRateDistortionError(
                "recodeable_floor_bytes must be non-negative when set"
            )
        if (
            self.recodeable_floor_bytes is not None
            and int(self.recodeable_floor_bytes) > int(self.bytes)
        ):
            raise LfPayloadRateDistortionError(
                "recodeable_floor_bytes cannot exceed current bytes"
            )


@dataclass(frozen=True)
class AtlasSensitivity:
    """One measured scorer-sensitivity cell, with its measurement scope.

    ``h_seg`` / ``h_pose`` are the atlas H_seg / H_pose for this (band, channel,
    orientation, frame-incidence, amplitude) cell — the EXACT d_seg the scorer
    suffers when this spectral content is perturbed at ``amplitude_lsb``. The scope
    fields are the validity envelope; an estimate that would use this cell outside
    its scope must fail closed.
    """

    band_index: int
    h_seg: float
    h_pose: float
    channel_basis: str = ""
    channel: str = ""
    orientation: str = ""
    frame_incidence: str = ""
    amplitude_lsb: float | None = None
    authority_tier: str = "exact_cpu_advisory"
    artifact_path: str = ""

    def __post_init__(self) -> None:
        if int(self.band_index) < 0:
            raise LfPayloadRateDistortionError("AtlasSensitivity.band_index must be >= 0")
        if float(self.h_seg) < 0.0 or float(self.h_pose) < 0.0:
            raise LfPayloadRateDistortionError(
                "AtlasSensitivity H_seg / H_pose must be non-negative"
            )
        if self.artifact_path and str(self.artifact_path).startswith("/tmp"):
            raise LfPayloadRateDistortionError(
                "AtlasSensitivity.artifact_path must be durable (not /tmp)"
            )


@dataclass(frozen=True)
class AtlasScope:
    """The measurement envelope of the atlas as a whole (its swept grid).

    Built from the atlas ``grid`` block. A coefficient-group coordinate OUTSIDE any
    of these sets is an extrapolation the planner refuses (scope_invalid).
    """

    band_indices: frozenset[int]
    channel_bases: frozenset[str]
    channels: frozenset[str]
    orientations: frozenset[str]
    frame_incidences: frozenset[str]
    amplitudes_lsb: tuple[float, ...]
    authority_tier: str = "exact_cpu_advisory"
    artifact_path: str = ""

    def amplitude_in_scope(self, amplitude_lsb: float | None) -> bool:
        """An amplitude is in scope when it equals a swept amplitude (within a tiny
        tolerance) OR sits inside the swept [min, max] interval. ``None`` (unspecified)
        is in scope only when the atlas swept at least one amplitude."""
        if not self.amplitudes_lsb:
            return False
        if amplitude_lsb is None:
            return True
        a = float(amplitude_lsb)
        lo, hi = min(self.amplitudes_lsb), max(self.amplitudes_lsb)
        tol = 1e-9 + 1e-6 * max(abs(lo), abs(hi), 1.0)
        if lo - tol <= a <= hi + tol:
            return True
        return any(abs(a - s) <= tol for s in self.amplitudes_lsb)


@dataclass(frozen=True)
class BaselineScoreTerms:
    """The receiver-surface baseline the planner deltas against (from G1b)."""

    d_seg: float
    d_pose: float
    archive_bytes: int
    axis_tag: str = "[macOS-CPU advisory]"

    def __post_init__(self) -> None:
        if float(self.d_seg) < 0.0 or float(self.d_pose) < 0.0:
            raise LfPayloadRateDistortionError("baseline d_seg / d_pose must be >= 0")
        if int(self.archive_bytes) < 0:
            raise LfPayloadRateDistortionError("baseline archive_bytes must be >= 0")


# ---------------------------------------------------------------------------
# Frame1 JOINT SAFE CONE input — the per-pixel spatial budget surface (#35 -> #46).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Frame1ConeMap:
    """The per-pixel frame1 JOINT SAFE CONE budget surface (from #35).

    This is the SPATIAL granularity the band×orientation atlas cannot resolve: for
    every frame1 pixel on the SegNet grid (384x512) the cone declares a perturbation
    budget (``joint_cone_radius``) such that coarsening a frame1-touching coefficient
    AT that pixel by <= radius leaves both contest scorers' response within tolerance.
    The ``fragile_cone_mask`` is the binding-constraint set (radius < half a uint8
    step) where NO frame1-touching byte may move.

    Built from the ``tac.optimization.frame1_joint_safe_cone.Frame1JointSafeCone``
    arrays — either in-memory (tests) or loaded from a ``cone_pair_*.npz`` map written
    by ``tools/build_frame1_joint_safe_cone.py --save-maps`` (the #35 CLI output;
    the .npz carries ``joint_cone_radius`` + ``fragile_cone_mask`` + companions).

    A section is refined by the cone ONLY when the section's content actually touches
    frame1 (``frame_incidence`` includes ``frame1_only`` or ``both_*``); a frame0-only
    section has no frame1 cone constraint and the masked action does not apply.
    """

    # (H, W) per-pixel joint cone radius in scorer input units [0, 255].
    joint_cone_radius: Any
    # (H, W) bool: True where the cone is fragile (radius < fragile threshold).
    fragile_cone_mask: Any
    # The cone's fragile-radius threshold (a pixel with radius >= this is "free").
    fragile_radius_threshold: float = 0.5
    # (H, W) OPTIONAL per-pixel joint P18/P19 sensitivity (from the cone's
    # ``joint_sensitivity`` array). When present, the masked action's distortion cost
    # is weighted by the fraction of the section's sensitivity that actually lives on
    # the FREE pixels (not just their pixel count) — capturing the cone's core claim
    # that the free pixels are the LOW-sensitivity ones. When absent, the masked
    # action falls back to the conservative pixel-count free fraction.
    joint_sensitivity: Any = None
    # Provenance / scope (carried through to every masked action for audit).
    source_path: str = ""
    axis_tag: str = "[macOS-CPU advisory]"

    def __post_init__(self) -> None:
        import numpy as _np

        radius = _np.asarray(self.joint_cone_radius, dtype=_np.float64)
        fragile = _np.asarray(self.fragile_cone_mask)
        if radius.ndim != 2:
            raise LfPayloadRateDistortionError(
                f"Frame1ConeMap.joint_cone_radius must be 2-D (H, W); got {radius.shape}"
            )
        if fragile.shape != radius.shape:
            raise LfPayloadRateDistortionError(
                "Frame1ConeMap.fragile_cone_mask shape must match joint_cone_radius"
            )
        if float(_np.nanmin(radius)) < 0.0:
            raise LfPayloadRateDistortionError(
                "Frame1ConeMap.joint_cone_radius must be non-negative"
            )
        if float(self.fragile_radius_threshold) < 0.0:
            raise LfPayloadRateDistortionError(
                "Frame1ConeMap.fragile_radius_threshold must be >= 0"
            )
        if self.joint_sensitivity is not None:
            js = _np.asarray(self.joint_sensitivity, dtype=_np.float64)
            if js.shape != radius.shape:
                raise LfPayloadRateDistortionError(
                    "Frame1ConeMap.joint_sensitivity shape must match joint_cone_radius"
                )
            if float(_np.nanmin(js)) < 0.0:
                raise LfPayloadRateDistortionError(
                    "Frame1ConeMap.joint_sensitivity must be non-negative"
                )
        if self.source_path and str(self.source_path).startswith("/tmp"):
            raise LfPayloadRateDistortionError(
                "Frame1ConeMap.source_path must be durable (not /tmp)"
            )
        # FAIL-CLOSED: an all-zero radius is the #35 "gradient not reachable" /
        # empty-cone signature. Refusing it here means a non-reachable cone can
        # never silently produce an all-permissive (everything-free) masked plan.
        if float(_np.abs(radius).sum()) <= 0.0:
            raise LfPayloadRateDistortionError(
                "Frame1ConeMap.joint_cone_radius is identically zero — the cone has "
                "NO free budget (or the upstream gradient was not reachable). Refusing "
                "to emit an all-permissive cone-masked plan; re-measure the cone."
            )

    @property
    def free_pixel_fraction(self) -> float:
        """Fraction of frame1 pixels with usable joint budget (radius >= threshold).

        This is the spatial fraction of a frame1-touching section's coefficients that
        the masked action may coarsen for free; the complement (the fragile set) is
        preserved at full precision."""
        import numpy as _np

        radius = _np.asarray(self.joint_cone_radius, dtype=_np.float64)
        thr = float(self.fragile_radius_threshold)
        return float((radius >= thr).mean())

    @property
    def fragile_pixel_fraction(self) -> float:
        import numpy as _np

        return float(_np.asarray(self.fragile_cone_mask).astype(bool).mean())

    @property
    def n_free_pixels(self) -> int:
        import numpy as _np

        radius = _np.asarray(self.joint_cone_radius, dtype=_np.float64)
        return int((radius >= float(self.fragile_radius_threshold)).sum())

    @property
    def n_pixels(self) -> int:
        import numpy as _np

        return int(_np.asarray(self.joint_cone_radius).size)

    @property
    def free_set_sensitivity_share(self) -> float | None:
        """Fraction of the cone's TOTAL joint sensitivity that lives on the FREE pixels.

        This is the cone's core distortion claim made concrete: the free pixels (high
        radius) are the LOW-sensitivity ones, so their share of the total sensitivity is
        SMALLER than their share of the pixel count (``free_pixel_fraction``). The
        masked action gives up only the distortion that lives on the free set, so the
        distortion cost is weighted by THIS share, not the pixel count. ``None`` when no
        ``joint_sensitivity`` map was supplied (fall back to the pixel-count fraction)."""
        if self.joint_sensitivity is None:
            return None
        import numpy as _np

        js = _np.asarray(self.joint_sensitivity, dtype=_np.float64)
        total = float(js.sum())
        if total <= 0.0:
            return 0.0
        free = self.free_mask()
        return float(js[free].sum() / total)

    def free_mask(self) -> Any:
        """Boolean ``(H, W)`` keep-coarse mask: True where the pixel is free to coarsen."""
        import numpy as _np

        radius = _np.asarray(self.joint_cone_radius, dtype=_np.float64)
        return radius >= float(self.fragile_radius_threshold)

    @classmethod
    def from_npz(
        cls,
        npz_path: str,
        *,
        fragile_radius_threshold: float | None = None,
        axis_tag: str = "[macOS-CPU advisory]",
    ) -> Frame1ConeMap:
        """Load a cone map from a ``cone_pair_*.npz`` (the #35 CLI ``--save-maps`` output).

        Reads the EXACT arrays the #35 CLI writes (``joint_cone_radius`` +
        ``fragile_cone_mask``); never invents the schema. ``fragile_radius_threshold``
        defaults to recovering the threshold from the fragile mask vs the radius when
        not supplied (so a map written at a non-default threshold round-trips)."""
        import numpy as _np

        p = str(npz_path)
        if p.startswith("/tmp"):
            raise LfPayloadRateDistortionError(
                "cone npz path must be durable (not /tmp)"
            )
        with _np.load(p) as z:
            files = set(z.files)
            if "joint_cone_radius" not in files:
                raise LfPayloadRateDistortionError(
                    f"cone npz {p!r} has no 'joint_cone_radius' array (got {sorted(files)}); "
                    "is this a tools/build_frame1_joint_safe_cone.py --save-maps output?"
                )
            radius = _np.asarray(z["joint_cone_radius"], dtype=_np.float64)
            if "fragile_cone_mask" in files:
                fragile = _np.asarray(z["fragile_cone_mask"]).astype(bool)
            else:
                # Derive from the radius using the (supplied or default) threshold.
                thr = float(
                    fragile_radius_threshold
                    if fragile_radius_threshold is not None
                    else 0.5
                )
                fragile = radius < thr
            # The #35 .npz also carries the per-pixel joint sensitivity; use it for the
            # sensitivity-share-weighted distortion model when present.
            joint_sens = (
                _np.asarray(z["joint_sensitivity"], dtype=_np.float64)
                if "joint_sensitivity" in files
                else None
            )
        # Recover the threshold from the data when not supplied: the smallest radius
        # not flagged fragile is the threshold floor; default to 0.5 (half a uint8 step).
        if fragile_radius_threshold is not None:
            thr = float(fragile_radius_threshold)
        else:
            free = radius[~fragile]
            thr = float(free.min()) if free.size else 0.5
        return cls(
            joint_cone_radius=radius,
            fragile_cone_mask=fragile,
            fragile_radius_threshold=thr,
            joint_sensitivity=joint_sens,
            source_path=p,
            axis_tag=str(axis_tag),
        )


def _section_touches_frame1(group: CoefficientGroup) -> bool:
    """A section is cone-refinable iff its coefficients touch frame1.

    SegNet reads ONLY frame1 and PoseNet reads frame1 through its frame1 channels,
    so a section whose ``frame_incidence`` is ``frame1_only`` or any ``both_*`` (both
    frames perturbed) touches frame1. An empty incidence (agnostic) is treated as
    touching frame1 (the conservative, fail-closed choice — the cone constraint is
    applied rather than skipped). A frame0-only section (``frame0_only``) has NO
    frame1 constraint and the cone does not refine it."""
    inc = str(group.frame_incidence)
    if inc == "":
        return True
    if inc.startswith("frame0"):
        return False
    return inc.startswith("frame1") or inc.startswith("both")


def estimate_mask_coding_cost_bytes(free_mask: Any) -> int:
    """Real coding cost (bytes) of the per-pixel keep/coarsen boolean mask.

    The mask MUST pay rent: it is a sidecar the receiver needs to know which pixels
    were coarsened. We MEASURE its cost (not guess): bit-pack the boolean mask then
    brotli quality=11 (CLAUDE.md L32 "brotli quality=11 max for sidecar"). This is an
    HONEST upper-ish bound on the mask's archive footprint; the exact re-measure
    refines it, but the planner already charges the mask its measured rent so a mask
    whose cost exceeds the bytes it frees is rejected by THE LAW. A spatially-coherent
    cone mask (large flat free regions) compresses well; a salt-and-pepper mask does
    not — and the cost correctly reflects that."""
    import numpy as _np

    try:
        import brotli  # type: ignore[import-not-found]
    except ModuleNotFoundError:  # pragma: no cover - brotli is a runtime dep
        # No brotli: fall back to a bit-packed RAW size (no entropy coding). This is a
        # strictly LARGER (more conservative) cost, so the mask still pays rent; the
        # ranking remains fail-closed (over-charging the mask never over-admits).
        m = _np.asarray(free_mask).astype(bool).ravel()
        return int(_np.packbits(m).nbytes)

    m = _np.asarray(free_mask).astype(bool).ravel()
    packed = _np.packbits(m).tobytes()
    coded = brotli.compress(packed, quality=11)
    return len(coded)


# ---------------------------------------------------------------------------
# The score-unit primitives (the LAW expressed exactly once).
# ---------------------------------------------------------------------------
def delta_pose_score(d_pose_base: float, d_pose_after: float) -> float:
    """Δ of the nonlinear pose term ``sqrt(10·d_pose)``.

    POSITIVE when the action makes pose WORSE (d_pose rises). The pose term is
    nonlinear, so this must be computed on the term, not on raw d_pose."""
    a = math.sqrt(_POSE_INNER * max(float(d_pose_base), 0.0))
    b = math.sqrt(_POSE_INNER * max(float(d_pose_after), 0.0))
    return b - a


def delta_distortion_score(
    d_seg_base: float,
    d_pose_base: float,
    d_seg_after: float,
    d_pose_after: float,
) -> float:
    """ΔS_distortion = 100·Δd_seg + Δsqrt(10·d_pose) (the non-rate half of the law).

    POSITIVE when the action makes distortion WORSE. Removing a section that buys
    distortion value raises d_seg/d_pose => positive ΔS_distortion (the cost of
    dropping). The LAW keeps the section iff this cost exceeds the rate it frees."""
    return _SEG_COEF * (float(d_seg_after) - float(d_seg_base)) + delta_pose_score(
        d_pose_base, d_pose_after
    )


def delta_rate_score(delta_bytes: int) -> float:
    """Δ of the rate term ``25·bytes/N``. NEGATIVE when the action frees bytes."""
    return _RATE_COEF * float(delta_bytes) / float(CONTEST_ARCHIVE_RATE_DENOM)


def keep_component(delta_distortion: float, delta_bytes_freed: int) -> bool:
    """THE LAW. ``keep iff -ΔS_distortion(drop) > 25·Δbytes/N``.

    ``delta_distortion`` is the ΔS_distortion of DROPPING the component (>=0 when
    dropping hurts distortion). ``delta_bytes_freed`` is the positive byte count
    the drop frees. Keep when the distortion saved by NOT dropping exceeds the rate
    a drop would free: ``delta_distortion > 25·delta_bytes_freed/N``.

    Equivalently to the prompt's ``-ΔS_distortion(c) > 25·Δbytes(c)/N``: here
    ΔS_distortion(c) is the distortion change of REMOVING c (positive), so
    ``-ΔS_distortion`` in the prompt's sign convention (where keeping is the
    reference) maps to the same threshold comparison."""
    if int(delta_bytes_freed) <= 0:
        # A keep/drop with no byte movement: keep iff dropping hurts distortion.
        return float(delta_distortion) > 0.0
    return float(delta_distortion) > delta_rate_score(int(delta_bytes_freed))


# ---------------------------------------------------------------------------
# Atlas adaptation.
# ---------------------------------------------------------------------------
def atlas_scope_from_grid(atlas: Mapping[str, Any]) -> AtlasScope:
    """Build the validity envelope from a ``scorer_spectral_sensitivity.v2`` atlas."""
    grid = dict(atlas.get("grid") or {})
    cells = list(atlas.get("cells") or [])
    if not grid and not cells:
        raise LfPayloadRateDistortionError(
            "atlas has neither a grid block nor cells; cannot derive scope"
        )
    n_bands = int(grid.get("n_bands", 0) or 0)
    band_indices = (
        frozenset(range(n_bands))
        if n_bands > 0
        else frozenset(int(c.get("band_index", 0)) for c in cells)
    )
    channel_bases = frozenset(str(x) for x in (grid.get("channel_bases") or [])) or (
        frozenset(str(c.get("channel_basis", "")) for c in cells)
    )
    channels = frozenset(
        str(x)
        for x in (
            list(grid.get("rgb_channels") or []) + list(grid.get("yuv_channels") or [])
        )
    ) or frozenset(str(c.get("channel", "")) for c in cells)
    orientations = frozenset(str(x) for x in (grid.get("orientations") or [])) or (
        frozenset(str(c.get("orientation", "")) for c in cells)
    )
    frame_incidences = frozenset(
        str(x) for x in (grid.get("frame_incidences") or [])
    ) or frozenset(str(c.get("frame_incidence", "")) for c in cells)
    amplitudes = tuple(float(x) for x in (grid.get("amplitudes_lsb") or [])) or tuple(
        sorted({float(c.get("amplitude_lsb", 0.0)) for c in cells})
    )
    return AtlasScope(
        band_indices=band_indices,
        channel_bases=channel_bases,
        channels=channels,
        orientations=orientations,
        frame_incidences=frame_incidences,
        amplitudes_lsb=amplitudes,
        authority_tier=str(atlas.get("authority_tier", "exact_cpu_advisory")),
        artifact_path=str((atlas.get("source_raw") or {}).get("path", "")),
    )


def atlas_sensitivities_from_cells(
    atlas: Mapping[str, Any],
) -> tuple[AtlasSensitivity, ...]:
    """Parse the atlas ``cells`` into typed ``AtlasSensitivity`` rows."""
    out: list[AtlasSensitivity] = []
    artifact = str((atlas.get("source_raw") or {}).get("path", ""))
    authority = str(atlas.get("authority_tier", "exact_cpu_advisory"))
    for c in atlas.get("cells") or []:
        out.append(
            AtlasSensitivity(
                band_index=int(c.get("band_index", 0)),
                h_seg=float(c.get("H_seg", c.get("d_seg_exact", 0.0)) or 0.0),
                h_pose=float(c.get("H_pose", 0.0) or 0.0),
                channel_basis=str(c.get("channel_basis", "")),
                channel=str(c.get("channel", "")),
                orientation=str(c.get("orientation", "")),
                frame_incidence=str(c.get("frame_incidence", "")),
                amplitude_lsb=(
                    float(c["amplitude_lsb"])
                    if c.get("amplitude_lsb") is not None
                    else None
                ),
                authority_tier=authority,
                artifact_path=artifact,
            )
        )
    return tuple(out)


def _coord_matches(want: str, have: str) -> bool:
    """An empty ``want`` matches any ``have`` (the section is agnostic on that axis)."""
    return want == "" or want == have


def _group_in_scope(group: CoefficientGroup, scope: AtlasScope) -> tuple[bool, str]:
    """Fail-closed scope check: every declared coordinate must be inside the envelope."""
    for b in group.band_indices:
        if int(b) not in scope.band_indices:
            return False, f"band_index {b} outside measured bands {sorted(scope.band_indices)}"
    if group.channel_basis and group.channel_basis not in scope.channel_bases:
        return False, f"channel_basis {group.channel_basis!r} not measured"
    if group.channel and group.channel not in scope.channels:
        return False, f"channel {group.channel!r} not measured"
    if group.orientation and group.orientation not in scope.orientations:
        return False, f"orientation {group.orientation!r} not measured"
    if group.frame_incidence and group.frame_incidence not in scope.frame_incidences:
        return False, f"frame_incidence {group.frame_incidence!r} not measured"
    if not scope.amplitude_in_scope(group.amplitude_lsb):
        return False, f"amplitude_lsb {group.amplitude_lsb!r} outside swept range"
    return True, ""


def _matched_cells(
    group: CoefficientGroup, sensitivities: Sequence[AtlasSensitivity]
) -> list[AtlasSensitivity]:
    """All atlas cells whose coordinates match the group (summed across its bands)."""
    matched: list[AtlasSensitivity] = []
    band_set = {int(b) for b in group.band_indices}
    for s in sensitivities:
        if int(s.band_index) not in band_set:
            continue
        if not _coord_matches(group.channel_basis, s.channel_basis):
            continue
        if not _coord_matches(group.channel, s.channel):
            continue
        if not _coord_matches(group.orientation, s.orientation):
            continue
        if not _coord_matches(group.frame_incidence, s.frame_incidence):
            continue
        if group.amplitude_lsb is not None and s.amplitude_lsb is not None:
            tol = 1e-9 + 1e-6 * max(abs(group.amplitude_lsb), 1.0)
            if abs(float(s.amplitude_lsb) - float(group.amplitude_lsb)) > tol:
                continue
        matched.append(s)
    return matched


@dataclass(frozen=True)
class SectionSensitivityEstimate:
    """The atlas-derived distortion value a section buys (or a scope refusal)."""

    section_name: str
    atlas_scope_valid: bool
    scope_reason: str
    est_d_seg_value: float | None
    """Σ H_seg over matched cells — the d_seg the scorer suffers if the section's
    content is removed/zeroed. ``None`` when scope-invalid."""
    est_d_pose_value: float | None
    """Σ H_pose over matched cells. ``None`` when scope-invalid."""
    matched_cell_count: int

    def to_row(self) -> dict[str, Any]:
        return {
            "section_name": self.section_name,
            "atlas_scope_valid": self.atlas_scope_valid,
            "scope_reason": self.scope_reason,
            "est_d_seg_value": self.est_d_seg_value,
            "est_d_pose_value": self.est_d_pose_value,
            "matched_cell_count": self.matched_cell_count,
        }


def estimate_section_sensitivity(
    section: PayloadSection,
    sensitivities: Sequence[AtlasSensitivity],
    scope: AtlasScope,
) -> SectionSensitivityEstimate:
    """Atlas-derived distortion value of a section, fail-closed outside scope."""
    in_scope, reason = _group_in_scope(section.coefficient_group, scope)
    if not in_scope:
        return SectionSensitivityEstimate(
            section_name=section.name,
            atlas_scope_valid=False,
            scope_reason=reason,
            est_d_seg_value=None,
            est_d_pose_value=None,
            matched_cell_count=0,
        )
    matched = _matched_cells(section.coefficient_group, sensitivities)
    if not matched:
        # In the swept grid but no cell matched the exact coordinate combination:
        # this is still an extrapolation (we have no measured datum for it).
        return SectionSensitivityEstimate(
            section_name=section.name,
            atlas_scope_valid=False,
            scope_reason="no atlas cell matched the section's coordinate combination",
            est_d_seg_value=None,
            est_d_pose_value=None,
            matched_cell_count=0,
        )
    return SectionSensitivityEstimate(
        section_name=section.name,
        atlas_scope_valid=True,
        scope_reason="",
        est_d_seg_value=float(sum(s.h_seg for s in matched)),
        est_d_pose_value=float(sum(s.h_pose for s in matched)),
        matched_cell_count=len(matched),
    )


# ---------------------------------------------------------------------------
# Candidate action evaluation (PROPOSAL rows).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CandidateActionEvaluation:
    """A PROPOSED action over a payload section, scored by the LAW (a PREDICTION).

    Distinct from ``evaluator_action_waterfill.CandidateActionEvaluation``: that
    one is the EXACT measured authority surface; THIS one is the atlas-estimate
    proposal that decides which sections to re-measure first. Every row carries the
    false-authority contract (``promotable=False`` / ``requires_exact_remeasure=True``).
    """

    action_id: str
    action_kind: str
    section_name: str
    est_delta_d_seg: float | None
    """Estimated Δd_seg the action causes (>=0 when it hurts seg). None if scope-invalid."""
    est_delta_d_pose: float | None
    """Estimated Δd_pose the action causes. None if scope-invalid."""
    delta_bytes: int
    """Δarchive_bytes (NEGATIVE when the action frees bytes)."""
    atlas_scope_valid: bool
    scope_reason: str
    baseline: BaselineScoreTerms

    def __post_init__(self) -> None:
        if self.action_kind not in _ACTION_KINDS:
            raise LfPayloadRateDistortionError(
                f"action_kind must be one of {_ACTION_KINDS}; got {self.action_kind!r}"
            )

    @property
    def est_delta_distortion_score(self) -> float | None:
        """ΔS_distortion = 100·Δd_seg + Δsqrt(10·d_pose) (None when scope-invalid)."""
        if (
            not self.atlas_scope_valid
            or self.est_delta_d_seg is None
            or self.est_delta_d_pose is None
        ):
            return None
        d_seg_after = max(self.baseline.d_seg + float(self.est_delta_d_seg), 0.0)
        d_pose_after = max(self.baseline.d_pose + float(self.est_delta_d_pose), 0.0)
        return delta_distortion_score(
            self.baseline.d_seg, self.baseline.d_pose, d_seg_after, d_pose_after
        )

    @property
    def est_delta_rate_score(self) -> float:
        return delta_rate_score(self.delta_bytes)

    @property
    def est_delta_score_total(self) -> float | None:
        """Total estimated ΔS = ΔS_distortion + ΔS_rate (None when scope-invalid).

        NEGATIVE => the action lowers the predicted contest score (admit candidate
        for exact re-measure)."""
        dd = self.est_delta_distortion_score
        if dd is None:
            return None
        return dd + self.est_delta_rate_score

    @property
    def value_per_byte(self) -> float | None:
        """Predicted score reduction per byte freed (the reverse-waterfill key).

        For a byte-FREEING action (delta_bytes < 0): ``-ΔS_total / |delta_bytes|``
        — higher is better (more predicted score reduction per byte freed). A
        byte-freeing action with NEGATIVE predicted ΔS_total has positive
        value_per_byte and ranks high. ``None`` when scope-invalid (cannot rank)."""
        total = self.est_delta_score_total
        if total is None:
            return None
        freed = -int(self.delta_bytes)
        if freed <= 0:
            # Action adds bytes: value-per-byte is the score reduction per added
            # byte (negative ΔS over positive added bytes). Rare for this planner.
            added = int(self.delta_bytes)
            if added <= 0:
                return None
            return -total / float(added)
        return -total / float(freed)

    @property
    def keep_section_under_law(self) -> bool | None:
        """THE LAW applied to a DROP of this section's full value.

        Keep iff the distortion the section buys exceeds the rate a full drop frees.
        Only meaningful for byte-freeing actions; ``None`` when scope-invalid."""
        dd = self.est_delta_distortion_score
        if dd is None:
            return None
        freed = -int(self.delta_bytes)
        # ΔS_distortion of dropping (dd) is >=0 when dropping hurts; keep iff it
        # exceeds the freed rate. For non-drop actions dd is the partial cost.
        return keep_component(dd, freed)

    @property
    def pays_rent_predicted(self) -> bool | None:
        """Predicted admission: the action lowers total predicted score.

        NONE when scope-invalid (the only honest verdict is "re-measure exactly")."""
        total = self.est_delta_score_total
        if total is None:
            return None
        return total < 0.0

    def to_row(self) -> dict[str, Any]:
        return {
            "schema": "snerv_lf_payload_candidate_action_proposal.v1",
            "action_id": self.action_id,
            "action_kind": self.action_kind,
            "section_name": self.section_name,
            "est_delta_d_seg": self.est_delta_d_seg,
            "est_delta_d_pose": self.est_delta_d_pose,
            "delta_bytes": int(self.delta_bytes),
            "est_delta_distortion_score": self.est_delta_distortion_score,
            "est_delta_rate_score": self.est_delta_rate_score,
            "est_delta_score_total": self.est_delta_score_total,
            "value_per_byte": self.value_per_byte,
            "keep_section_under_law": self.keep_section_under_law,
            "pays_rent_predicted": self.pays_rent_predicted,
            "atlas_scope_valid": self.atlas_scope_valid,
            "scope_reason": self.scope_reason,
            "baseline_d_seg": self.baseline.d_seg,
            "baseline_d_pose": self.baseline.d_pose,
            "baseline_archive_bytes": int(self.baseline.archive_bytes),
            # The false-authority contract (this is a PREDICTION, not a score).
            "authority": "planning_control_false_authority",
            "axis_tag": self.baseline.axis_tag,
            "score_claim": False,
            "promotion_eligible": False,
            "promotable": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "requires_exact_remeasure": True,
        }


# ---------------------------------------------------------------------------
# Action builders.
# ---------------------------------------------------------------------------
def _quantize_fraction(quantize_step: float) -> tuple[float, float]:
    """Map a quantization step Δ to (bytes_kept_fraction, value_kept_fraction).

    A coarser step frees more bytes and gives up more distortion value. We use a
    simple, MONOTONE, dependency-free model: a step Δ over a unit-amplitude section
    keeps ``1/(1+Δ)`` of the bytes and ``1/(1+Δ)`` of the distortion value (both
    decline with Δ; the value declines because coarser coefficients perturb the
    scorer less). This is an ESTIMATE only; the exact re-measure replaces it. The
    point of the planner is RANKING, and the ranking is monotone in Δ regardless of
    the exact functional form."""
    step = max(float(quantize_step), 0.0)
    frac = 1.0 / (1.0 + step)
    return frac, frac


def build_drop_action(
    section: PayloadSection,
    estimate: SectionSensitivityEstimate,
    baseline: BaselineScoreTerms,
) -> CandidateActionEvaluation:
    """DROP: remove the whole section. Frees all its bytes; gives up all its value."""
    if estimate.atlas_scope_valid:
        d_seg = estimate.est_d_seg_value
        d_pose = estimate.est_d_pose_value
    else:
        d_seg = None
        d_pose = None
    return CandidateActionEvaluation(
        action_id=f"{section.name}::drop",
        action_kind=ACTION_DROP,
        section_name=section.name,
        est_delta_d_seg=d_seg,
        est_delta_d_pose=d_pose,
        delta_bytes=-int(section.bytes),
        atlas_scope_valid=estimate.atlas_scope_valid,
        scope_reason=estimate.scope_reason,
        baseline=baseline,
    )


def build_quantize_action(
    section: PayloadSection,
    estimate: SectionSensitivityEstimate,
    baseline: BaselineScoreTerms,
    quantize_step: float,
) -> CandidateActionEvaluation:
    """QUANTIZE: coarsen by Δ; frees a byte fraction, gives up a value fraction."""
    bytes_kept, value_kept = _quantize_fraction(quantize_step)
    freed = round(section.bytes * (1.0 - bytes_kept))
    if estimate.atlas_scope_valid:
        # Value GIVEN UP = (1 - value_kept) × the section's full value.
        d_seg = (estimate.est_d_seg_value or 0.0) * (1.0 - value_kept)
        d_pose = (estimate.est_d_pose_value or 0.0) * (1.0 - value_kept)
    else:
        d_seg = None
        d_pose = None
    return CandidateActionEvaluation(
        action_id=f"{section.name}::quantize_delta={quantize_step:g}",
        action_kind=ACTION_QUANTIZE,
        section_name=section.name,
        est_delta_d_seg=d_seg,
        est_delta_d_pose=d_pose,
        delta_bytes=-freed,
        atlas_scope_valid=estimate.atlas_scope_valid,
        scope_reason=estimate.scope_reason,
        baseline=baseline,
    )


def build_recode_action(
    section: PayloadSection,
    estimate: SectionSensitivityEstimate,
    baseline: BaselineScoreTerms,
) -> CandidateActionEvaluation | None:
    """RECODE: lossless re-encode to ``recodeable_floor_bytes``; ~zero distortion cost.

    Returns ``None`` when the section declares no recodeable floor (no free-bytes
    opportunity to propose). A lossless recode gives up zero distortion value
    (est_delta_d_seg/pose = 0), so it ALWAYS pays rent when it frees bytes — the
    free-bytes branch of the waterfill."""
    if section.recodeable_floor_bytes is None:
        return None
    freed = int(section.bytes) - int(section.recodeable_floor_bytes)
    if freed <= 0:
        return None
    # Lossless recode: scope validity does not gate it (no distortion estimate
    # needed — the value given up is exactly zero by construction).
    return CandidateActionEvaluation(
        action_id=f"{section.name}::recode",
        action_kind=ACTION_RECODE,
        section_name=section.name,
        est_delta_d_seg=0.0,
        est_delta_d_pose=0.0,
        delta_bytes=-freed,
        atlas_scope_valid=True,
        scope_reason="lossless recode (zero distortion cost by construction)",
        baseline=baseline,
    )


@dataclass(frozen=True)
class ConeMaskedActionAccounting:
    """The byte + distortion accounting of a cone-masked quantize action (audit surface).

    Exposed so a reviewer can see EXACTLY why a masked action's net bytes / distortion
    came out as they did (CLAUDE.md "Max observability" — decomposable per signal)."""

    free_pixel_fraction: float
    """Spatial fraction of the section's coefficients on coarsenable (free) pixels."""
    gross_bytes_freed: int
    """Bytes freed by coarsening the free fraction at step Δ (before mask rent)."""
    mask_coding_cost_bytes: int
    """The per-pixel keep/coarsen mask's OWN coding cost (bit-pack + brotli q=11)."""
    net_bytes_freed: int
    """gross_bytes_freed - mask_coding_cost_bytes (the mask pays rent)."""
    value_kept_fraction: float
    """Fraction of the coarsened coefficients' distortion value retained at step Δ."""
    distortion_weight: float
    """The weight applied to the section's atlas distortion value = the free set's
    SENSITIVITY share (when a joint_sensitivity map is supplied) or the pixel-count
    free fraction (fallback). This is < free_pixel_fraction when sensitivity
    concentrates on the preserved fragile set — the cone's structural advantage."""
    used_sensitivity_share: bool
    """True when distortion_weight came from the joint_sensitivity map (the cone's
    spatial sensitivity weighting); False when it fell back to the pixel-count fraction."""
    quantize_step: float
    fragile_radius_threshold: float
    n_free_pixels: int
    n_pixels: int

    def to_row(self) -> dict[str, Any]:
        return {
            "free_pixel_fraction": self.free_pixel_fraction,
            "gross_bytes_freed": int(self.gross_bytes_freed),
            "mask_coding_cost_bytes": int(self.mask_coding_cost_bytes),
            "net_bytes_freed": int(self.net_bytes_freed),
            "value_kept_fraction": self.value_kept_fraction,
            "distortion_weight": self.distortion_weight,
            "used_sensitivity_share": self.used_sensitivity_share,
            "quantize_step": self.quantize_step,
            "fragile_radius_threshold": self.fragile_radius_threshold,
            "n_free_pixels": int(self.n_free_pixels),
            "n_pixels": int(self.n_pixels),
        }


def build_cone_masked_quantize_action(
    section: PayloadSection,
    estimate: SectionSensitivityEstimate,
    baseline: BaselineScoreTerms,
    cone: Frame1ConeMap,
    quantize_step: float,
) -> tuple[CandidateActionEvaluation, ConeMaskedActionAccounting] | None:
    """SPATIALLY-MASKED quantize: coarsen the section by Δ ONLY at the cone-FREE pixels.

    This is the #35 -> #46 wiring: instead of coarsening the whole section uniformly
    (:func:`build_quantize_action`), coarsen only at frame1 pixels whose joint cone
    radius >= the fragile threshold (the spatially-free set), preserving full precision
    on the fragile set. The cone gives the per-pixel granularity the band×orientation
    atlas cannot resolve.

    Byte accounting (the mask pays rent):
      * gross_bytes_freed = section.bytes * f * (Δ / (1 + Δ))
            where ``f`` = cone free-pixel fraction (the coarsenable spatial share).
      * mask_coding_cost = real bit-packed + brotli q=11 size of the keep/coarsen mask.
      * delta_bytes = -(gross_bytes_freed - mask_coding_cost)   (NEGATIVE = frees bytes)
      A mask whose coding cost exceeds the bytes it frees yields net_bytes_freed <= 0;
      the action then ADDS bytes (delta_bytes >= 0) and THE LAW rejects it — exactly
      the prompt's "a mask whose bytes exceed its savings is rejected".

    Distortion accounting (cone-radius-weighted):
      The distortion given up is the section's atlas value scaled by the FREE fraction
      and the coarsen ratio: ``est_delta = section_value * f * (1 - value_kept)``. Only
      the free-set coefficients are coarsened, and the free set is BY CONSTRUCTION the
      low-sensitivity (high-radius) pixels — so the fragile (high-sensitivity) pixels
      contribute ~0 to the given-up distortion because they are preserved. This is the
      structural advantage over the unmasked action: same coarsen ratio, but the
      distortion cost is paid only on the spatially-free pixels.

    Returns ``None`` when the section does not touch frame1 (no cone constraint), when
    the cone has no free pixels for this section, or when the section is scope-invalid
    AND we therefore cannot estimate the distortion (the masked action's whole point is
    the cone-weighted distortion estimate; a scope-invalid section is segregated to the
    needs_exact_remeasure path by the plain quantize/drop builders).
    """
    if not _section_touches_frame1(section.coefficient_group):
        return None
    if not estimate.atlas_scope_valid:
        # Scope-invalid: the cone-weighted distortion estimate would extrapolate the
        # atlas value. Refuse (fail-closed) — the plain drop/quantize builders already
        # route this section to needs_exact_remeasure.
        return None

    step = max(float(quantize_step), 0.0)
    f = float(cone.free_pixel_fraction)
    if f <= 0.0:
        # No coarsenable pixel: the cone is all-fragile for this section -> nothing to do.
        return None
    bytes_kept, value_kept = _quantize_fraction(step)
    coarsen_ratio = 1.0 - bytes_kept  # = Δ/(1+Δ)
    # Bytes freed are proportional to the FREE PIXEL COUNT fraction (the spatial share
    # of coefficients coarsened).
    gross_freed = round(int(section.bytes) * f * coarsen_ratio)
    mask_cost = estimate_mask_coding_cost_bytes(cone.free_mask())
    net_freed = gross_freed - mask_cost

    # Distortion given up: the section's atlas value, scaled by the share of the
    # section's sensitivity that lives on the FREE set (NOT the pixel count). The
    # cone's core claim is that the free pixels are LOW-sensitivity, so their
    # sensitivity share is < their pixel-count share — meaning the masked action
    # gives up DISPROPORTIONATELY less distortion per byte freed than the unmasked
    # action (which pays the full section value). When no joint_sensitivity map is
    # supplied, fall back to the conservative pixel-count fraction ``f`` (no claimed
    # advantage — the honest default that never over-admits).
    sens_share = cone.free_set_sensitivity_share
    dist_weight = float(sens_share) if sens_share is not None else f
    seg_value = float(estimate.est_d_seg_value or 0.0)
    pose_value = float(estimate.est_d_pose_value or 0.0)
    value_given_up = dist_weight * (1.0 - value_kept)
    d_seg = seg_value * value_given_up
    d_pose = pose_value * value_given_up

    accounting = ConeMaskedActionAccounting(
        free_pixel_fraction=f,
        gross_bytes_freed=gross_freed,
        mask_coding_cost_bytes=mask_cost,
        net_bytes_freed=net_freed,
        value_kept_fraction=value_kept,
        distortion_weight=dist_weight,
        used_sensitivity_share=sens_share is not None,
        quantize_step=step,
        fragile_radius_threshold=float(cone.fragile_radius_threshold),
        n_free_pixels=cone.n_free_pixels,
        n_pixels=cone.n_pixels,
    )
    action = CandidateActionEvaluation(
        action_id=f"{section.name}::quantize_cone_masked_delta={step:g}",
        action_kind=ACTION_QUANTIZE_CONE_MASKED,
        section_name=section.name,
        est_delta_d_seg=d_seg,
        est_delta_d_pose=d_pose,
        # NEGATIVE frees bytes; when the mask costs more than it frees this is >= 0
        # (the action ADDS bytes) and THE LAW rejects it (pays_rent_predicted=False).
        delta_bytes=-net_freed,
        atlas_scope_valid=True,
        scope_reason=(
            f"cone-masked: coarsen Δ={step:g} on {f:.3f} free frame1 pixels "
            f"(mask rent {mask_cost} B vs gross {gross_freed} B freed)"
        ),
        baseline=baseline,
    )
    return action, accounting


# ---------------------------------------------------------------------------
# The planner.
# ---------------------------------------------------------------------------
def plan_lf_payload_actions(
    sections: Sequence[PayloadSection],
    sensitivities: Sequence[AtlasSensitivity],
    scope: AtlasScope,
    baseline: BaselineScoreTerms,
    *,
    quantize_steps: Sequence[float] = (0.5, 1.0, 2.0),
    frame1_cone_map: Frame1ConeMap | None = None,
    cone_quantize_steps: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Rank candidate actions over the payload by predicted value-per-byte (THE LAW).

    For each section, propose DROP (if droppable), QUANTIZE at each step, and RECODE
    (if a recode floor is declared). Rank scope-valid, rent-paying actions by
    descending ``value_per_byte`` (most predicted score reduction per byte freed
    first). Scope-invalid actions are segregated into ``needs_exact_remeasure`` and
    NEVER ranked above scope-valid ones (fail-closed).

    SPATIAL refinement (#35 -> #46; gated, default-OFF): when ``frame1_cone_map`` is
    supplied (the per-pixel frame1 JOINT SAFE CONE budget from
    ``tools/build_frame1_joint_safe_cone.py``), ALSO propose a SPATIALLY-MASKED
    quantize per frame1-touching section at each ``cone_quantize_steps`` step (defaults
    to ``quantize_steps``). The masked action coarsens ONLY the cone-free frame1 pixels
    and charges the keep/coarsen mask its OWN coding rent (the mask must pay rent under
    THE LAW). The cone gives the spatial granularity the band×orientation atlas cannot
    resolve. When ``frame1_cone_map is None`` the plan is byte-identical to before
    (backward-compatible)."""

    cone_steps = (
        tuple(quantize_steps)
        if cone_quantize_steps is None
        else tuple(cone_quantize_steps)
    )

    estimates: dict[str, SectionSensitivityEstimate] = {}
    proposals: list[CandidateActionEvaluation] = []
    cone_accounting: dict[str, dict[str, Any]] = {}
    for section in sections:
        est = estimate_section_sensitivity(section, sensitivities, scope)
        estimates[section.name] = est
        if section.droppable:
            proposals.append(build_drop_action(section, est, baseline))
        for step in quantize_steps:
            proposals.append(build_quantize_action(section, est, baseline, step))
        recode = build_recode_action(section, est, baseline)
        if recode is not None:
            proposals.append(recode)
        if frame1_cone_map is not None:
            for step in cone_steps:
                masked = build_cone_masked_quantize_action(
                    section, est, baseline, frame1_cone_map, step
                )
                if masked is not None:
                    action, accounting = masked
                    proposals.append(action)
                    cone_accounting[action.action_id] = accounting.to_row()

    ranked: list[CandidateActionEvaluation] = []
    not_paying: list[CandidateActionEvaluation] = []
    needs_remeasure: list[CandidateActionEvaluation] = []
    for p in proposals:
        if not p.atlas_scope_valid or p.value_per_byte is None:
            needs_remeasure.append(p)
        elif p.pays_rent_predicted:
            ranked.append(p)
        else:
            not_paying.append(p)

    ranked.sort(
        key=lambda p: (p.value_per_byte if p.value_per_byte is not None else -math.inf),
        reverse=True,
    )

    total_predicted_freed = sum(-p.delta_bytes for p in ranked if p.delta_bytes < 0)
    total_predicted_delta_score = sum(
        p.est_delta_score_total
        for p in ranked
        if p.est_delta_score_total is not None
    )

    plan: dict[str, Any] = {
        "schema": "snerv_lf_payload_rd_plan.v1",
        "baseline": {
            "d_seg": baseline.d_seg,
            "d_pose": baseline.d_pose,
            "archive_bytes": int(baseline.archive_bytes),
            "axis_tag": baseline.axis_tag,
        },
        "law": "keep c iff -dS_distortion(c) > 25*dBytes(c)/37545489",
        "section_sensitivity_estimates": [
            estimates[s.name].to_row() for s in sections
        ],
        "ranked_actions": [p.to_row() for p in ranked],
        "not_paying_rent": [p.to_row() for p in not_paying],
        "needs_exact_remeasure": [p.to_row() for p in needs_remeasure],
        "n_ranked": len(ranked),
        "n_not_paying": len(not_paying),
        "n_needs_remeasure": len(needs_remeasure),
        "best_action_id": ranked[0].action_id if ranked else None,
        "best_value_per_byte": ranked[0].value_per_byte if ranked else None,
        "total_predicted_bytes_freed": int(total_predicted_freed),
        "total_predicted_delta_score": total_predicted_delta_score,
        "requires_recompute_after_accept": True,
        "note": (
            "PROPOSAL surface only. ranked_actions are PREDICTIONS from the measured "
            "scorer atlas; each MUST be applied + exactly re-measured into a "
            "tac.optimization.evaluator_action_waterfill.CandidateActionEvaluation "
            "before any admission. needs_exact_remeasure rows are scope-invalid: the "
            "atlas cannot estimate them without extrapolating."
        ),
        "authority": "planning_control_false_authority",
        "axis_tag": baseline.axis_tag,
        "score_claim": False,
        "promotion_eligible": False,
        "promotable": False,
        "ready_for_exact_eval_dispatch": False,
    }
    # Cone refinement provenance (only present when the cone was supplied).
    if frame1_cone_map is not None:
        plan["frame1_cone"] = {
            "active": True,
            "source_path": frame1_cone_map.source_path,
            "free_pixel_fraction": frame1_cone_map.free_pixel_fraction,
            "fragile_pixel_fraction": frame1_cone_map.fragile_pixel_fraction,
            "fragile_radius_threshold": frame1_cone_map.fragile_radius_threshold,
            "n_free_pixels": frame1_cone_map.n_free_pixels,
            "n_pixels": frame1_cone_map.n_pixels,
            "cone_quantize_steps": list(cone_steps),
            "axis_tag": frame1_cone_map.axis_tag,
            "n_cone_masked_actions": len(cone_accounting),
            "cone_masked_accounting": cone_accounting,
        }
    else:
        plan["frame1_cone"] = {"active": False}
    return plan


__all__ = [
    "ACTION_DROP",
    "ACTION_QUANTIZE",
    "ACTION_QUANTIZE_CONE_MASKED",
    "ACTION_RECODE",
    "CONTEST_BYTE_PRICE",
    "AtlasScope",
    "AtlasSensitivity",
    "BaselineScoreTerms",
    "CandidateActionEvaluation",
    "CoefficientGroup",
    "ConeMaskedActionAccounting",
    "Frame1ConeMap",
    "LfPayloadRateDistortionError",
    "PayloadSection",
    "SectionSensitivityEstimate",
    "atlas_scope_from_grid",
    "atlas_sensitivities_from_cells",
    "build_cone_masked_quantize_action",
    "build_drop_action",
    "build_quantize_action",
    "build_recode_action",
    "delta_distortion_score",
    "delta_pose_score",
    "delta_rate_score",
    "estimate_mask_coding_cost_bytes",
    "estimate_section_sensitivity",
    "keep_component",
    "plan_lf_payload_actions",
]
