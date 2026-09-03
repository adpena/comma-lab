# SPDX-License-Identifier: MIT
"""Canonical provenance law for the MarginBandSatisficing threshold.

The threshold is not an independently tunable constant:

    m_safe = headroom * delta_R

``delta_R`` is read from ``reports/delta_R_noise_floor_n600.json`` (ddm_dr1, 2026-09-04, all
600 pairs; the 2026-07-12 n96 artifact ``reports/delta_R_noise_floor.json`` is HISTORICAL — its
contiguous prefix understated the annulus noise floor by 11.70%, see the m88 instance).  The default
headroom is the smallest integer multiple whose threshold covers the same
artifact's measured full-R annulus p95.  With the current artifact this is
DERIVED ``ceil(0.03887428045272823 / 0.021881818771362305) = 2`` and therefore
DERIVED ``m_safe = 0.04376363754272461`` (was 0.039180326461791926 on the n96 prefix).  A factor of 3 remains a legitimate
future treatment, but no measured A/B currently pins it as the default.

The fallback values below are a class-4 WAIVER copied exactly from the current
MEASURED artifact.  They exist only so local config compilation remains
possible when the report is unavailable; every fallback use is recorded by
``LawRef.resolve`` and must be re-derived when the artifact is regenerated.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    VERIFIED_VIA_EMPIRICAL_ANCHOR,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import build_provenance_for_research_sidecar
from tac.witness_dsl.lawref import (
    LADDER_DERIVED_LIVE,
    InputRef,
    LawRef,
    resolve,
)

EQUATION_ID = "margin_band_satisficing_threshold_v1"
DELTA_R_ARTIFACT = "reports/delta_R_noise_floor_n600.json"
DELTA_R_ARTIFACT_N96_HISTORICAL = "reports/delta_R_noise_floor.json"

# WAIVER fallbacks copied exactly from reports/delta_R_noise_floor_n600.json on
# 2026-09-04 (ddm_dr1).  They are not new measurements or guessed round numbers.
FALLBACK_DELTA_R = 0.021881818771362305
FALLBACK_FULL_R_ANNULUS_P95 = 0.03887428045272823
FALLBACK_N_FRAMES = 600
FALLBACK_ANNULUS_BAND = 1.0

# ASSUMED numerical tolerance for checking two deterministic fp64 arithmetic
# paths.  This is not a scientific or score threshold.
INVARIANT_FP_TOL = 1e-12

_AXIS = "[macOS-CPU advisory . frozen CPU-torch SegNet . NON-PROMOTABLE]"


def _resolved_artifact_path(artifact_path: str | Path) -> Path:
    path = Path(artifact_path)
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parents[3] / path


def _artifact_mtime_utc(artifact_path: str | Path) -> str:
    """Return MEASURED filesystem UTC because the source report embeds no clock."""

    timestamp = _resolved_artifact_path(artifact_path).stat().st_mtime
    return datetime.fromtimestamp(timestamp, tz=UTC).isoformat().replace(
        "+00:00", "Z"
    )


def margin_safe_threshold(delta_r: float, headroom: float) -> float:
    """Return DERIVED ``m_safe = headroom * delta_R`` with strict domain checks."""

    delta = float(delta_r)
    factor = float(headroom)
    if not math.isfinite(delta) or delta <= 0.0:
        raise ValueError(f"delta_r must be finite and > 0, got {delta_r!r}")
    if not math.isfinite(factor) or factor < 1.0:
        raise ValueError(f"headroom must be finite and >= 1, got {headroom!r}")
    return factor * delta


def minimum_integer_headroom(delta_r: float, full_r_annulus_p95: float) -> float:
    """Smallest integer factor whose threshold covers measured full-R annulus p95."""

    delta = float(delta_r)
    full_r = float(full_r_annulus_p95)
    if not math.isfinite(full_r) or full_r <= 0.0:
        raise ValueError(
            "full_r_annulus_p95 must be finite and > 0, "
            f"got {full_r_annulus_p95!r}"
        )
    # ``margin_safe_threshold`` owns the delta-domain validation.
    margin_safe_threshold(delta, 1.0)
    return float(max(1, math.ceil(full_r / delta)))


def _artifact_values(
    artifact_path: str | Path,
) -> tuple[float, float, int, float, bool]:
    """Read the two MEASURED anchors, or return the documented WAIVER fallback."""

    path = _resolved_artifact_path(artifact_path)
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return (
            FALLBACK_DELTA_R,
            FALLBACK_FULL_R_ANNULUS_P95,
            FALLBACK_N_FRAMES,
            FALLBACK_ANNULUS_BAND,
            True,
        )

    try:
        delta = float(doc["delta_R"])
        full_r = float(doc["cross_check_full_R_vs_gt_direct"]["annulus"]["p95"])
        n_frames = int(doc["n_frames"])
        annulus_band = float(doc["band"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"MarginBandSatisficing provenance artifact has invalid schema: {path}"
        ) from exc
    margin_safe_threshold(delta, 1.0)
    minimum_integer_headroom(delta, full_r)
    if n_frames <= 0:
        raise ValueError(f"n_frames must be > 0 in provenance artifact: {path}")
    if not math.isfinite(annulus_band) or annulus_band <= 0.0:
        raise ValueError(f"band must be finite and > 0 in provenance artifact: {path}")
    return delta, full_r, n_frames, annulus_band, False


def margin_safe_lawref(
    *, headroom: float, artifact_path: str | Path = DELTA_R_ARTIFACT
) -> LawRef:
    """Build the DERIVED-LIVE LawRef consumed by the DSL factory."""

    factor = float(headroom)
    # Validate before constructing a fallback so a bad config cannot hide
    # behind the artifact-missing waiver.
    margin_safe_threshold(FALLBACK_DELTA_R, factor)
    return LawRef(
        equation_id=EQUATION_ID,
        inputs={
            "delta_r": InputRef.anchor(
                str(artifact_path),
                "delta_R",
                "MEASURED p95 uint8-at-camera SegNet-margin perturbation over the annulus; "
                "tools/measure_delta_R_noise_floor.py -> reports/delta_R_noise_floor_n600.json",
            ),
            "headroom": InputRef.literal(
                factor,
                "DERIVED-AT-CONFIG R-survival multiplier; default is the smallest integer "
                "covering the measured full-R annulus p95",
            ),
        },
        ladder_class=LADDER_DERIVED_LIVE,
        fallback=margin_safe_threshold(FALLBACK_DELTA_R, factor),
        fallback_waiver_reason=(
            "reports/delta_R_noise_floor_n600.json unavailable; use the exact 2026-09-04 "
            "MEASURED n600 delta_R anchor until the artifact is restored"
        ),
    )


@dataclass(frozen=True)
class MarginBandThresholdResolution:
    """Resolved config values plus machine-checkable provenance state."""

    delta_r: float
    full_r_annulus_p95: float
    n_frames: int
    annulus_band: float
    headroom: float
    m_safe: float
    artifact_path: str
    artifact_fallback_used: bool
    lawref_fallback_used: bool
    lawref_manifest: dict


def resolve_margin_band_threshold(
    *,
    headroom: float | None = None,
    artifact_path: str | Path = DELTA_R_ARTIFACT,
    repo_root: str | Path | None = None,
) -> MarginBandThresholdResolution:
    """Resolve MEASURED ``delta_R`` and DERIVED ``m_safe`` for DSL compilation.

    ``headroom=None`` derives the default from the artifact's full-R p95.  An
    explicit headroom is an AT-CONFIG treatment value, but ``m_safe`` remains
    derived by this canonical law and cannot become an independent literal.
    """

    (
        delta_from_artifact,
        full_r_p95,
        n_frames,
        annulus_band,
        artifact_fallback,
    ) = _artifact_values(artifact_path)
    factor = (
        minimum_integer_headroom(delta_from_artifact, full_r_p95)
        if headroom is None
        else float(headroom)
    )
    lawref = margin_safe_lawref(headroom=factor, artifact_path=artifact_path)
    resolved = resolve(lawref, repo_root=repo_root)

    delta = FALLBACK_DELTA_R if resolved.fallback_used else float(
        next(record.value for record in resolved.resolved_inputs if record.name == "delta_r")
    )
    expected = margin_safe_threshold(delta, factor)
    if not math.isclose(
        float(resolved.value),
        expected,
        rel_tol=INVARIANT_FP_TOL,
        abs_tol=INVARIANT_FP_TOL,
    ):
        raise ValueError(
            "MarginBandSatisficing canonical-law invariant failed: "
            f"m_safe={resolved.value!r}, headroom={factor!r}, delta_r={delta!r}"
        )
    return MarginBandThresholdResolution(
        delta_r=delta,
        full_r_annulus_p95=full_r_p95,
        n_frames=n_frames,
        annulus_band=annulus_band,
        headroom=factor,
        m_safe=float(resolved.value),
        artifact_path=str(artifact_path),
        artifact_fallback_used=artifact_fallback,
        lawref_fallback_used=resolved.fallback_used,
        lawref_manifest=resolved.to_dict(),
    )


def build_margin_band_satisficing_threshold_v1() -> CanonicalEquation:
    """Build the canonical threshold law with its MEASURED artifact anchor."""

    resolution = resolve_margin_band_threshold()
    artifact_utc = _artifact_mtime_utc(DELTA_R_ARTIFACT)
    provenance = build_provenance_for_research_sidecar(
        sidecar_path=DELTA_R_ARTIFACT,
        reactivation_criteria=(
            "re-run tools/measure_delta_R_noise_floor.py on the required cohort; derive the "
            "minimum headroom again; A/B headroom 2 versus 3 before promoting 3"
        ),
        measurement_axis=_AXIS,
        hardware_substrate="apple_macos_cpu_torch",
        captured_at_utc=artifact_utc,
    )
    anchor = EmpiricalAnchor(
        anchor_id="margin_band_delta_r_noise_floor_n600_20260904",
        measurement_utc=artifact_utc,
        inputs={
            "artifact": DELTA_R_ARTIFACT,
            "measurement_time_source": (
                "MEASURED artifact filesystem mtime; source JSON embeds no UTC field"
            ),
            "measurement_tool": "tools/measure_delta_R_noise_floor.py",
            "n_frames": resolution.n_frames,
            "annulus_band": resolution.annulus_band,
            "supersedes": (
                "margin_band_delta_r_noise_floor_n96_20260708 (HISTORICAL; the n96 contiguous "
                "prefix understated the annulus p95 by 11.70%, ddm_dr1 2026-09-04)"
            ),
        },
        predicted_output={
            "law": "m_safe = headroom * delta_R",
            "headroom_policy": (
                "smallest integer factor covering measured full-R annulus p95"
            ),
        },
        empirical_output={
            "delta_R": resolution.delta_r,
            "full_R_annulus_p95": resolution.full_r_annulus_p95,
            "derived_headroom": resolution.headroom,
            "derived_m_safe": resolution.m_safe,
            "headroom_3_status": "OPEN_UNMEASURED_TREATMENT_NOT_DEFAULT",
            "treatment_delta_s": "UNMEASURED",
        },
        residual=0.0,
        source_artifact=DELTA_R_ARTIFACT,
        measurement_method=(
            "frozen CPU-torch SegNet margin perturbation; delta_R is annulus p95 of "
            "uint8-at-camera isolation; full-R p95 is the conservative coverage cross-check"
        ),
        provenance=provenance,
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
        noise_floor=0.0,
        noise_floor_provenance="deterministic arithmetic on the measured JSON anchor",
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="Margin-band satisficing R-survival threshold",
        one_line_summary=(
            "m_safe=headroom*delta_R; default headroom is the smallest integer covering "
            "the measured full-R annulus p95 (currently 2), while delta_R comes from the "
            "measured noise-floor artifact."
        ),
        latex_form=r"m_{\mathrm{safe}}=h_R\,\delta_R",
        python_callable_module_path=(
            "tac.canonical_equations.margin_band_satisficing_threshold_20260712:"
            "margin_safe_threshold"
        ),
        domain_of_validity={
            "included": [
                "MarginBandSatisficing config compilation",
                "SegNet signed-margin units measured by the delta_R artifact",
            ],
            "excluded": [
                "claim that headroom 3 improves d_seg",
                "score, promotion, or pointer movement",
                "artifact-free use without a recorded fallback waiver",
            ],
            "authority": _AXIS,
        },
        units_in={"delta_R": "SegNet logit-margin perturbation", "headroom": "dimensionless"},
        units_out={"m_safe": "SegNet signed logit margin"},
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={"threshold_arithmetic": 0.0},
        last_calibration_utc=artifact_utc,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.witness_dsl.curriculum_dsl.MarginBandSatisficing",
            "tac.witness_dsl.lawref.resolve",
        ),
        canonical_producers=("tools.measure_delta_R_noise_floor",),
        provenance=provenance,
    )


def populate_margin_band_satisficing_threshold_equation(
    *, path=None, lock_path=None, agent: str | None = None, subagent_id: str | None = None
) -> CanonicalEquation:
    """Append the equation through the locked, append-only registry helper."""

    from tac.canonical_equations.registry import register_canonical_equation

    equation = build_margin_band_satisficing_threshold_v1()
    register_canonical_equation(
        equation,
        path=path,
        lock_path=lock_path,
        agent=agent,
        subagent_id=subagent_id,
        notes=(
            "MarginBandSatisficing provenance repair: delta_R MEASURED anchor; "
            "headroom 2 DERIVED; m_safe DERIVED-LIVE; headroom-3 A/B remains open"
        ),
    )
    return equation


__all__ = [
    "DELTA_R_ARTIFACT",
    "DELTA_R_ARTIFACT_N96_HISTORICAL",
    "EQUATION_ID",
    "FALLBACK_ANNULUS_BAND",
    "FALLBACK_DELTA_R",
    "FALLBACK_FULL_R_ANNULUS_P95",
    "FALLBACK_N_FRAMES",
    "INVARIANT_FP_TOL",
    "MarginBandThresholdResolution",
    "build_margin_band_satisficing_threshold_v1",
    "margin_safe_lawref",
    "margin_safe_threshold",
    "minimum_integer_headroom",
    "populate_margin_band_satisficing_threshold_equation",
    "resolve_margin_band_threshold",
]
