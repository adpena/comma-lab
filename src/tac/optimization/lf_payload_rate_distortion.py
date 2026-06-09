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

_ACTION_KINDS = (ACTION_DROP, ACTION_QUANTIZE, ACTION_RECODE)


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
) -> dict[str, Any]:
    """Rank candidate actions over the payload by predicted value-per-byte (THE LAW).

    For each section, propose DROP (if droppable), QUANTIZE at each step, and RECODE
    (if a recode floor is declared). Rank scope-valid, rent-paying actions by
    descending ``value_per_byte`` (most predicted score reduction per byte freed
    first). Scope-invalid actions are segregated into ``needs_exact_remeasure`` and
    NEVER ranked above scope-valid ones (fail-closed).
    """

    estimates: dict[str, SectionSensitivityEstimate] = {}
    proposals: list[CandidateActionEvaluation] = []
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

    return {
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


__all__ = [
    "ACTION_DROP",
    "ACTION_QUANTIZE",
    "ACTION_RECODE",
    "CONTEST_BYTE_PRICE",
    "AtlasScope",
    "AtlasSensitivity",
    "BaselineScoreTerms",
    "CandidateActionEvaluation",
    "CoefficientGroup",
    "LfPayloadRateDistortionError",
    "PayloadSection",
    "SectionSensitivityEstimate",
    "atlas_scope_from_grid",
    "atlas_sensitivities_from_cells",
    "build_drop_action",
    "build_quantize_action",
    "build_recode_action",
    "delta_distortion_score",
    "delta_pose_score",
    "delta_rate_score",
    "estimate_section_sensitivity",
    "keep_component",
    "plan_lf_payload_actions",
]
