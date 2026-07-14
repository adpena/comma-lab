# SPDX-License-Identifier: MIT
"""Equal-budget basis-selection law grounded by the settled owed-16 n600 rows."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    VERIFIED_VIA_EMPIRICAL_ANCHOR,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import (
    build_provenance_for_predicted,
    build_provenance_for_research_sidecar,
)

EQUATION_ID = "optimal_basis_equal_budget_through_r_v1"
_UTC = "2026-07-14T12:50:00Z"
_VERDICT = ".omx/research/owed16_verdict_20260710.json"
_VERDICT_V2 = ".omx/research/owed16v2_verdict_20260710.json"
_DAG = ".omx/research/optimal_basis_beyond_fourier_DAG_FEED_20260714.md"


@dataclass(frozen=True)
class BasisMeasurement:
    family: str
    d_seg: float
    trainable_values: int
    archive_bytes: int | None
    n_pairs: int
    through_r: bool
    evidence_axis: str
    verdict_scope: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


MEASURED_EP675_ROWS = (
    BasisMeasurement(
        family="polar_directional_fourier_self_orient_off",
        d_seg=0.004244,
        trainable_values=109_559,
        archive_bytes=None,
        n_pairs=600,
        through_r=True,
        evidence_axis="[macOS-CPU advisory]",
        verdict_scope="bounded warm-start ep675, seed0, formulation only",
    ),
    BasisMeasurement(
        family="self_oriented_fourier_along8",
        d_seg=0.004259,
        trainable_values=111_095,
        archive_bytes=None,
        n_pairs=600,
        through_r=True,
        evidence_axis="[macOS-CPU advisory]",
        verdict_scope="bounded warm-start ep675, seed0, formulation only",
    ),
    BasisMeasurement(
        family="self_oriented_fourier_along26",
        d_seg=0.004286,
        trainable_values=111_095,
        archive_bytes=None,
        n_pairs=600,
        through_r=True,
        evidence_axis="[macOS-CPU advisory]",
        verdict_scope="bounded warm-start ep675, seed0, formulation only",
    ),
)


def select_basis_under_equal_budget(
    rows: tuple[BasisMeasurement, ...] = MEASURED_EP675_ROWS,
    *,
    max_trainable_values: int | None = None,
    max_archive_bytes: int | None = None,
) -> BasisMeasurement:
    r"""Return ``argmin_B d_seg(SegNet(R(G_B)))`` under recorded budgets.

    Missing archive bytes never pass an explicit archive-byte constraint.  This
    prevents the parameter-count result from being laundered into a byte-closed
    archive verdict.
    """

    if not rows:
        raise ValueError("at least one basis measurement is required")
    eligible: list[BasisMeasurement] = []
    for row in rows:
        if row.n_pairs != 600 or not row.through_r:
            continue
        if max_trainable_values is not None and row.trainable_values > max_trainable_values:
            continue
        if (
            max_archive_bytes is not None
            and (row.archive_bytes is None or row.archive_bytes > max_archive_bytes)
        ):
            continue
        eligible.append(row)
    if not eligible:
        raise ValueError("no n600 through-R basis row satisfies the requested custody/budget constraints")
    return min(eligible, key=lambda row: (row.d_seg, row.trainable_values, row.family))


def build_optimal_basis_equal_budget_through_r_v1() -> CanonicalEquation:
    winner = select_basis_under_equal_budget(max_trainable_values=111_095)
    anchor = EmpiricalAnchor(
        anchor_id="owed16_equal_parameter_basis_selection_ep675_20260710",
        measurement_utc="2026-07-10T15:31:20Z",
        inputs={
            "rows": [row.to_dict() for row in MEASURED_EP675_ROWS],
            "constraint": "n600 through-R and trainable_values <= 111095",
        },
        predicted_output={
            "historical_hypothesis": "self-oriented directional basis reduces d_seg by 48 percent"
        },
        empirical_output={
            "winner": winner.to_dict(),
            "delta_self_orient_along8_minus_off": 0.000015,
            "delta_self_orient_along26_minus_off": 0.000042,
            "directional_minus48_reconfirmed": False,
            "different_frame_verdict": "NO_VERDICT: no genuine localized frame row",
            "archive_byte_verdict": "NO_VERDICT: no byte-closed archives in owed16",
        },
        residual=0.48,
        source_artifact=_VERDICT_V2,
        measurement_method="settled matched ep675 saved-artifact re-derivation",
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
        provenance=build_provenance_for_research_sidecar(
            _VERDICT_V2,
            reactivation_criteria=(
                "do not rerun bounded warm-start along8/along26; reactivate only a distinct fresh-start "
                "or genuinely localized equal-budget family"
            ),
            measurement_axis="[macOS-CPU advisory]",
            hardware_substrate="apple_m5_max_cpu_torch",
            captured_at_utc="2026-07-10T15:31:20Z",
        ),
    )
    law_sha = hashlib.sha256(
        b"argmin basis dseg through R subject to parameter and archive byte budgets"
    ).hexdigest()
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="Equal-budget realized-through-R witness basis selection",
        one_line_summary=(
            "Choose the lowest real-n600 through-R d_seg basis row under explicit parameter and "
            "archive-byte budgets; missing byte custody is ineligible for a byte-constrained verdict."
        ),
        latex_form=(
            r"B^*(K,A)=\arg\min_{B\in\mathcal B}\ d_{seg}(\operatorname{SegNet}(R(G_{\theta_B,B})))"
            r"\quad\mathrm{s.t.}\quad |\theta_B|\le K,\ |A_B|\le A"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.optimal_basis_selection_20260714:select_basis_under_equal_budget"
        ),
        domain_of_validity={
            "vehicle": "level-set witness owed16 bounded warm-start formulation",
            "n_pairs": 600,
            "surface": "actual R plus frozen CPU-torch SegNet argmax",
            "axis": "[macOS-CPU advisory], non-promotable",
            "measured_families": [row.family for row in MEASURED_EP675_ROWS],
            "excluded_claims": (
                "no contest score, no archive-byte comparison, no fresh-start family verdict, "
                "no true curvelet/shearlet/wavelet verdict"
            ),
            "metric_duality": (
                "task500 metric_id=argmax_native_vjp_fidelity_v1 supplies G_q; a future "
                "metric-sparse basis minimizes the pullback Gram off-diagonal energy and measured "
                "through-R debt under the same budgets; n600 selection is NO-VERDICT_DATA_CUSTODY"
            ),
        },
        units_in={
            "d_seg": "argmax disagreement fraction",
            "trainable_values": "scalar values",
            "archive_bytes": "bytes or missing",
        },
        units_out={"basis": "stable family ID"},
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={"historical_minus48_vs_owed16": 0.48},
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.witness_dsl.optimal_basis_20260714",
            "task500 metric provider via BasisMetricInterface",
        ),
        canonical_producers=(_VERDICT, _VERDICT_V2, _DAG),
        provenance=build_provenance_for_predicted(
            model_id=EQUATION_ID,
            inputs_sha256=law_sha,
            measurement_axis="[derived selection law]",
            hardware_substrate="numpy-portable",
            captured_at_utc=_UTC,
        ),
    )


def populate_optimal_basis_equal_budget_through_r_v1(
    *, path=None, lock_path=None, agent: str | None = None, subagent_id: str | None = None
) -> CanonicalEquation:
    """Append through the locked registry helper only when explicitly invoked."""

    from tac.canonical_equations.registry import register_canonical_equation

    equation = build_optimal_basis_equal_budget_through_r_v1()
    register_canonical_equation(
        equation,
        path=path,
        lock_path=lock_path,
        agent=agent,
        subagent_id=subagent_id,
        notes="task497 saved-artifact equal-budget basis-selection law; pointer unchanged",
    )
    return equation


__all__ = [
    "EQUATION_ID",
    "MEASURED_EP675_ROWS",
    "BasisMeasurement",
    "build_optimal_basis_equal_budget_through_r_v1",
    "populate_optimal_basis_equal_budget_through_r_v1",
    "select_basis_under_equal_budget",
]
