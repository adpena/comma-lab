# SPDX-License-Identifier: MIT
"""Canonical equation: ``dsl_custodied_scalar_identity_v1`` — NON-DERIVATIONAL value custody.

The #332/#351 flag-custody law (CLAUDE.md "2026-07-14 catalog amendments — V9 provenance
is the anti-fake boundary"): a V9 semantic flag whose value is MEASURED or class-4 WAIVED
(a hardcoded config knob with typed ``HardcodedWaiverCustody``) is custodied through this
registered identity law — it *preserves bool(0/1)/int/float/string value bytes* and is
**explicitly non-derivational; it cannot manufacture scientific authority**.  The honest
rung lives on the LawRef's ``ladder_class`` (``hardcoded_waiver`` for the 2026-07-17
backfill) and in the per-flag provenance table, never in this equation.

Realized by the 2026-07-17 #332 backfill (``spec_v9_cgauge.attach_flag_custody``): every
semantic flag of the closed live V9 factory set now carries exactly one Lever owner, one
LawRef, one canonical compiler record (same equation_id), one runtime receipt schema, and
a value-provenance rung — MEASURED 3,862 -> 0 bijection residuals with deterministic
per-factory bijection hashes (apparatus row; NOT a score; pointer UNMOVED).

means != ends: this is APPARATUS (the anti-fake boundary), never goal progress.
"""
from __future__ import annotations

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

IDENTITY_EQUATION_ID = "dsl_custodied_scalar_identity_v1"

_BACKFILL_MEMO = ".omx/research/catalog332_flag_custody_backfill_20260717.md"

# MEASURED (2026-07-17, this backfill's before/after checker runs):
BACKFILL_RESIDUALS_BEFORE = 3862
BACKFILL_RESIDUALS_AFTER = 0


def custodied_scalar_identity(value):
    """value_out == value_in — the whole law.  No derivation is (or can be) claimed."""

    return value


def build_dsl_custodied_scalar_identity_v1() -> CanonicalEquation:
    """Build the registered non-derivational identity-custody canonical equation."""

    anchor_backfill = EmpiricalAnchor(
        anchor_id="catalog332_flag_custody_backfill_residuals_3862_to_0_20260717",
        measurement_utc="2026-07-17T00:00:00Z",
        inputs={
            "checker": "tac.preflight.check_config_flag_provenance_bijection_complete",
            "factories": [
                "v9_cgauge_432",
                "v9_cgauge_truly_optimal_core",
                "v9_cgauge_ideal_mod19",
                "v9_cgauge_ideal_mod32",
            ],
            "mechanism": "spec_v9_cgauge.attach_flag_custody (value-neutral rollup Lever + "
                         "compiler-record LawRef reconstruction + class-4 identity waivers)",
        },
        predicted_output={"residuals_after": 0},
        empirical_output={
            "residuals_before": BACKFILL_RESIDUALS_BEFORE,
            "residuals_after": BACKFILL_RESIDUALS_AFTER,
            "deterministic_bijection_hashes": True,
            "argv_byte_identity": "asserted per factory (attach-time fail-closed)",
        },
        residual=0.0,
        source_artifact=_BACKFILL_MEMO,
        measurement_method="live_checker_before_after_run_2026_07_17",
        empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_BACKFILL_MEMO,
            reactivation_criteria=(
                "per-flag waiver retirement: registering an executable derivation law or a "
                "content-hashed measured anchor for a flag replaces its identity custody"
            ),
            measurement_axis="[apparatus]",
            hardware_substrate="m5_max_cpu",
        ),
    )
    return CanonicalEquation(
        equation_id=IDENTITY_EQUATION_ID,
        name="Non-derivational DSL scalar identity custody (#332/#351 anti-fake boundary)",
        one_line_summary=(
            "value_out == value_in for MEASURED/WAIVED constants (bool 0/1, int, float, "
            "str bytes preserved); explicitly non-derivational — it cannot manufacture "
            "scientific authority"
        ),
        latex_form=r"\mathrm{custody}(v) = v \quad (\text{no derivation claimed})",
        python_callable_module_path=(
            "tac.canonical_equations.dsl_custodied_scalar_identity_20260717:"
            "custodied_scalar_identity"
        ),
        domain_of_validity={
            "role": "value custody ONLY (measured anchors / class-4 hardcoded waivers)",
            "vehicle": ["v9_cgauge closed factory set"],
            "forbidden": "citing this law as a derivation; it grants no mechanism, score, "
                         "or authority (CLAUDE.md #351)",
        },
        units_in={"value": "the flag's own units (opaque; bytes preserved)"},
        units_out={"value": "identical to units_in"},
        empirical_anchors=(anchor_backfill,),
        predicted_vs_empirical_residual={"backfill_residuals_after": 0.0},
        last_calibration_utc="2026-07-17T00:00:00Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.witness_dsl.spec_v9_cgauge.attach_flag_custody",
            "tac.v9_provenance_gates.check_config_flag_provenance_bijection_complete",
        ),
        canonical_producers=(
            "tac.canonical_equations.evaluators.eval_dsl_custodied_scalar_identity",
        ),
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_BACKFILL_MEMO,
            reactivation_criteria=(
                "identity custody is per-flag retirable: a registered derivation law or "
                "content-hashed measured anchor supersedes the class-4 waiver"
            ),
            measurement_axis="[apparatus]",
            hardware_substrate="m5_max_cpu",
        ),
    )


__all__ = [
    "BACKFILL_RESIDUALS_AFTER",
    "BACKFILL_RESIDUALS_BEFORE",
    "IDENTITY_EQUATION_ID",
    "build_dsl_custodied_scalar_identity_v1",
    "custodied_scalar_identity",
]
