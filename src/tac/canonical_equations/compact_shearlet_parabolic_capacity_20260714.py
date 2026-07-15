# SPDX-License-Identifier: MIT
"""Canonical equation for the genuine compact-shearlet parabolic capacity law.

This closes the registration debt left when
``tac.boundary_math.compact_shearlet_frame`` landed.  The equation deliberately
keeps structural frame certification separate from family selection: neither
the primitive's certificate nor the newer compiled-frame structural receipt is
a through-R score or an equal-byte family verdict.
"""

from __future__ import annotations

import math

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    VERIFIED_VIA_EMPIRICAL_ANCHOR,
    VERIFIED_VIA_SOURCE_INSPECTION,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

EQUATION_ID = "compact_shearlet_parabolic_capacity_v1"
MEASUREMENT_UTC = "2026-07-15T02:00:00Z"
AXIS = "MEASURED_LOCAL_CPU_NUMPY_FP64_STRUCTURAL_NOT_SCORE"
SOURCE_MODULE = "src/tac/boundary_math/compact_shearlet_frame.py"
SOURCE_CALLABLE_MODULE = "tac.boundary_math.compact_shearlet_frame"
STRUCTURAL_PROOF = (
    ".omx/research/"
    "genuine_curvelet_shearlet_structural_proof_v2_polar_frequency_wedge_20260714.json"
)
STRUCTURAL_PROOF_SHA256 = (
    "677a2252c43c1272ec0e2e83d65ce1b82d23b8ddb089d73a111a5f0b26d46d25"
)
STRUCTURAL_PROOF_COMPILED_SOURCE_SHA256 = (
    "2946f0a7647f03223d56b7921042a58e0a934825b95da9395f43d0aff98e46ef"
)
PRIMITIVE_SOURCE_SHA256_AT_REGISTRATION = (
    "1cfdeecd2eeb28406ef9be0e526a04e1af061e3d1a65d762589426f6253d543e"
)
SELECTION_STATUS = "NO_VERDICT_DATA_CUSTODY"


def compact_shearlet_sigma_pair(
    j: int,
    *,
    w0: float = 0.5,
    width_ratio: float = 2.0,
    aniso: float = 1.0,
    min_sigma: float = 0.02,
) -> tuple[float, float]:
    """Return the primitive's normal/tangent parabolic scale pair.

    Before clamping, ``sigma_n=w0*r^-j`` and
    ``sigma_t=aniso*w0*r^(-j/2)``, hence
    ``sigma_n=(sigma_t/aniso)^2/w0``.  The clamp is part of the finite-grid
    implementation and means the algebraic equality is scoped to scales above
    ``min_sigma``.
    """

    if isinstance(j, bool) or not isinstance(j, int) or j < 0:
        raise ValueError(f"j must be a non-negative int, got {j!r}")
    values = {
        "w0": float(w0),
        "width_ratio": float(width_ratio),
        "aniso": float(aniso),
        "min_sigma": float(min_sigma),
    }
    for name, value in values.items():
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and > 0, got {value!r}")
    if values["width_ratio"] <= 1.0:
        raise ValueError("width_ratio must exceed 1")
    if values["aniso"] < 1.0:
        raise ValueError("aniso must be >= 1")
    sigma_n = max(
        values["w0"] * values["width_ratio"] ** (-j), values["min_sigma"]
    )
    sigma_t = max(
        values["aniso"]
        * values["w0"]
        * values["width_ratio"] ** (-0.5 * j),
        values["min_sigma"],
    )
    return sigma_n, sigma_t


def compact_shearlet_structural_certificate_law() -> dict:
    """Run the primitive's genuine-shear/localization swap-test certificate."""

    from tac.boundary_math.compact_shearlet_frame import (
        CompactShearletConfig,
        shearlet_certificate,
    )

    return shearlet_certificate(CompactShearletConfig()).to_dict()


def _source_provenance():
    return build_provenance_for_research_sidecar(
        sidecar_path=SOURCE_MODULE,
        reactivation_criteria=(
            "re-run the structural certificate after any primitive change; a family "
            "selection additionally requires equal-byte parse-back and through-R custody"
        ),
        measurement_axis=AXIS,
        hardware_substrate="macOS arm64 local CPU NumPy-fp64 structural fixture",
        captured_at_utc=MEASUREMENT_UTC,
    )


def _proof_provenance():
    return build_provenance_for_research_sidecar(
        sidecar_path=STRUCTURAL_PROOF,
        reactivation_criteria=(
            "bind a selected compiled frame to exact archive bytes and measure through-R; "
            "structural proof alone cannot select a family"
        ),
        measurement_axis=AXIS,
        hardware_substrate="macOS arm64 local CPU NumPy-fp64 structural proof",
        captured_at_utc=MEASUREMENT_UTC,
    )


def build_compact_shearlet_parabolic_capacity_v1() -> CanonicalEquation:
    """Build the structural/parabolic equation without asserting a family winner."""

    certificate = compact_shearlet_structural_certificate_law()
    source_anchor = EmpiricalAnchor(
        anchor_id="compact_shearlet_primitive_swap_test_20260715",
        measurement_utc=MEASUREMENT_UTC,
        inputs={
            "primitive": SOURCE_CALLABLE_MODULE,
            "source_sha256": PRIMITIVE_SOURCE_SHA256_AT_REGISTRATION,
            "test": "shear/localization/rotation/Fourier swap certificate",
            "axis": AXIS,
        },
        predicted_output={
            "genuine_shear_steering": True,
            "spatially_localized": True,
            "parabolic_scaling_monotone": True,
        },
        empirical_output={
            **certificate,
            "score_claim": False,
            "promotion_eligible": False,
            "selection_metric_status": SELECTION_STATUS,
        },
        residual=0.0 if certificate["passes"] else 1.0,
        source_artifact=SOURCE_MODULE,
        measurement_method=(
            "deterministic primitive-owned certificate: shear anchor invariance vs matched "
            "rotation, integer-lattice preservation, localization vs Fourier, and "
            "parabolic-scale monotonicity"
        ),
        provenance=_source_provenance(),
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
    )
    proof_anchor = EmpiricalAnchor(
        anchor_id="compact_shearlet_separate_compiled_frame_structural_proof_20260714",
        measurement_utc=MEASUREMENT_UTC,
        inputs={
            "proof_sha256": STRUCTURAL_PROOF_SHA256,
            "compiled_source_sha256": STRUCTURAL_PROOF_COMPILED_SOURCE_SHA256,
            "primitive_source_sha256": PRIMITIVE_SOURCE_SHA256_AT_REGISTRATION,
            "same_source_bytes": False,
        },
        predicted_output={
            "use_as_separate_structural_context_only": True,
            "family_selection_allowed": False,
        },
        empirical_output={
            "proof_status": "COMPLETE",
            "compact_shearlet_structural_passed": True,
            "selection_metric_status": SELECTION_STATUS,
            "equal_archive_bytes_claim": False,
            "proven_frame_bounds": False,
            "score_claim": False,
            "promotion_eligible": False,
        },
        residual=0.0,
        source_artifact=STRUCTURAL_PROOF,
        measurement_method=(
            "source-inspected structural proof receipt; source hash is deliberately kept "
            "separate from the registered primitive rather than inferred equivalent"
        ),
        provenance=_proof_provenance(),
        empirical_verification_status=VERIFIED_VIA_SOURCE_INSPECTION,
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="Compact-shearlet genuine-shear parabolic capacity law",
        one_line_summary=(
            "The landed primitive is genuinely shear-steered, localized, and parabolically "
            "scaled; this is structural capacity evidence, not a family or score verdict."
        ),
        latex_form=(
            r"\sigma_n(j)=w_0r^{-j},\quad "
            r"\sigma_t(j)=a w_0r^{-j/2}\Rightarrow "
            r"\sigma_n=(\sigma_t/a)^2/w_0;\quad "
            r"S_k=\begin{bmatrix}1&k\\0&1\end{bmatrix},\ \det S_k=1"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.compact_shearlet_parabolic_capacity_20260714:"
            "compact_shearlet_sigma_pair"
        ),
        domain_of_validity={
            "primitive": SOURCE_CALLABLE_MODULE,
            "structural_scope": (
                "finite compact cone-adapted shear-steered frame; parabolic law above the "
                "finite-grid min_sigma clamp"
            ),
            "capacity_link": (
                "shearlet_nterm_upper_bounds_task_rate_v1 owns the cartoon-model "
                "approximation-rate upper bound; tightness remains unproved"
            ),
            "separate_compiled_proof_source": STRUCTURAL_PROOF_COMPILED_SOURCE_SHA256,
            "registered_primitive_source": PRIMITIVE_SOURCE_SHA256_AT_REGISTRATION,
            "source_equivalence_claim": False,
            "selection_metric_status": SELECTION_STATUS,
            "dsl_wire_status": "OWED_SERIALIZED_NO_LIVE_CONSUMER_ASSERTED",
            "research_only": True,
            "score_claim": False,
            "promotion_eligible": False,
            "excluded": (
                "frame tightness/completeness, exact transform inversion, equal-byte family "
                "ranking, through-R distortion, or score movement"
            ),
            "verdict_scope": (
                "structural/parabolic law for the named primitive only; all representation "
                "families remain open under selection-metric and byte custody"
            ),
        },
        units_in={
            "j": "nonnegative_scale_index",
            "w0": "normalized_coordinate_units",
            "width_ratio": "dimensionless",
            "aniso": "dimensionless",
        },
        units_out={
            "sigma_n": "normalized_coordinate_units",
            "sigma_t": "normalized_coordinate_units",
        },
        empirical_anchors=(source_anchor, proof_anchor),
        predicted_vs_empirical_residual={
            "primitive_structural_certificate": 0.0 if certificate["passes"] else 1.0,
            "separate_proof_structural_status": 0.0,
        },
        last_calibration_utc=MEASUREMENT_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(),
        canonical_producers=(
            SOURCE_CALLABLE_MODULE,
            "tac.canonical_equations.deepmath_amortizing_argmax_laws_20260704",
        ),
        provenance=_source_provenance(),
    )


def populate_compact_shearlet_parabolic_capacity_v1(
    *, path=None, lock_path=None, agent=None, subagent_id=None
) -> CanonicalEquation:
    """Append the equation through the canonical locked registry API."""

    from tac.canonical_equations.registry import register_canonical_equation

    equation = build_compact_shearlet_parabolic_capacity_v1()
    register_canonical_equation(
        equation,
        path=path,
        lock_path=lock_path,
        agent=agent,
        subagent_id=subagent_id,
        notes=(
            "task #504 closes compact-shearlet equation-ledger gap; structural only; "
            "selection NO_VERDICT_DATA_CUSTODY; DSL wire owed"
        ),
    )
    return equation


__all__ = [
    "AXIS",
    "EQUATION_ID",
    "MEASUREMENT_UTC",
    "PRIMITIVE_SOURCE_SHA256_AT_REGISTRATION",
    "SELECTION_STATUS",
    "SOURCE_CALLABLE_MODULE",
    "SOURCE_MODULE",
    "STRUCTURAL_PROOF",
    "STRUCTURAL_PROOF_COMPILED_SOURCE_SHA256",
    "STRUCTURAL_PROOF_SHA256",
    "build_compact_shearlet_parabolic_capacity_v1",
    "compact_shearlet_sigma_pair",
    "compact_shearlet_structural_certificate_law",
    "populate_compact_shearlet_parabolic_capacity_v1",
]
