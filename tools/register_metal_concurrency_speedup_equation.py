# SPDX-License-Identifier: MIT
# no-argparse-OK: no argv consumed — __main__ registers one pinned equation from measured constants
"""One-shot registration of canonical equation ``metal_concurrency_speedup_gv1_v1``.

EQUATIONS-LEG for ddm_gv1 (operator 2026-09-04: "Remember we can fully saturate cpu, gpu, and
ane").  The governance question the governor must answer before admitting a second Metal cell is
NOT "is there a free GPU" but "does the machine get MORE WORK DONE".  This equation is that
quantity, and its null hypothesis is the honest one:

    H0 (the null the governor must beat):  total_steps_per_min(N) = serial_baseline
    admit a further cell  <=>  total_steps_per_min(N) >= serial_baseline

MEASURED anchor (ddm_gv1, 2026-09-04, M5 Max, MLX/Metal, two concurrent QBR1 fairform cells):

    concurrency N=1  total 28.0    steps/min   [SECOND-HAND: measured by MAIN, window unrecorded]
    concurrency N=2  total 31.2854 steps/min   [ddm_gv1, 420.004 s window, 108 + 111 steps]

    speedup S(2) = 31.2854 / 28.0 = 1.1173
    per-cell efficiency = S(2) / 2 = 0.5586  (each cell runs at ~55.9% of its solo rate)

READ THIS BEFORE CITING IT.  This is ONE concurrent observation against ONE second-hand serial
observation.  It is NOT a fitted scaling law, there is no N=3 point, and no noise floor has been
measured (a single 420 s window gives no variance estimate).  What it licenses is exactly one
decision -- "at N=2 on this box, a second cell PAID by 11.7%" -- and it must be recalibrated the
moment a second anchor exists.  Extrapolating S(N) beyond N=2 from this row would be the
constants-are-poison failure: the per-cell efficiency 0.5586 is already below 1/N=0.5's neighbour,
so the marginal cell at N=3 could plausibly be net-negative and NOTHING here predicts it.

Non-promotable: wall-clock throughput is not a contest score.  ``score_claim=False`` everywhere;
the axis is ``[macOS-Metal wall-clock]``, never a contest axis.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.canonical_equations import (  # noqa: E402
    RECALIBRATE_ON_NEW_ANCHORS,
    CanonicalEquation,
    EmpiricalAnchor,
    get_equation_by_id,
    register_canonical_equation,
)
from tac.provenance.builders import (  # noqa: E402
    build_provenance_for_predicted,
    build_provenance_for_research_sidecar,
)

EQUATION_ID = "metal_concurrency_speedup_gv1_v1"

#: The serial baseline the null hypothesis uses (steps/min at concurrency 1).
SERIAL_BASELINE_STEPS_PER_MIN = 28.0

#: The measured concurrent total at N=2 (steps/min), ddm_gv1 420.004 s window.
CONCURRENT_TOTAL_STEPS_PER_MIN_N2 = 31.2854

CONTENTION_LEDGER = ".omx/state/metal_contention_ledger.jsonl"
MEMO = ".omx/research/ddm_gv1_governor_memory_guard_controller_polish_20260904.md"


def concurrency_speedup(total_steps_per_min: float, serial_baseline_steps_per_min: float) -> float:
    """Speedup of the whole machine under concurrency. ``>= 1.0`` means the extra cell PAID.

    This is the exact predicate ``tools/cell_admission.throughput_verdict`` gates on.
    """
    if serial_baseline_steps_per_min <= 0.0:
        raise ValueError("serial baseline must be positive")
    return float(total_steps_per_min) / float(serial_baseline_steps_per_min)


def per_cell_efficiency(total_steps_per_min: float, serial_baseline: float, n_cells: int) -> float:
    """Fraction of its solo rate each cell retains under concurrency (1.0 = perfect scaling)."""
    if n_cells <= 0:
        raise ValueError("n_cells must be positive")
    return concurrency_speedup(total_steps_per_min, serial_baseline) / float(n_cells)


def _inputs_sha256(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, allow_nan=False).encode("utf-8")
    ).hexdigest()


def build_equation() -> CanonicalEquation:
    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    anchor_inputs = {
        "concurrency": 2,
        "serial_baseline_steps_per_min": SERIAL_BASELINE_STEPS_PER_MIN,
        "window_s": 420.004,
        "cells": {
            "seed_20260902_area_cap_control_native100": 15.4284,
            "seed_20260902_tau_band_control_native100": 15.8570,
        },
        "steps_observed": {"area_cap": 108, "tau_band": 111},
    }
    anchor = EmpiricalAnchor(
        anchor_id="ddm_gv1_metal_concurrency_n2_20260904",
        measurement_utc="2026-09-04T15:55:37Z",
        inputs=anchor_inputs,
        # The NULL: concurrency buys nothing, the machine stays at its serial rate.
        predicted_output=SERIAL_BASELINE_STEPS_PER_MIN,
        empirical_output=CONCURRENT_TOTAL_STEPS_PER_MIN_N2,
        residual=CONCURRENT_TOTAL_STEPS_PER_MIN_N2 - SERIAL_BASELINE_STEPS_PER_MIN,
        source_artifact=CONTENTION_LEDGER,
        measurement_method=(
            "two reads of each live cell's history.jsonl row count separated by a 420.004 s "
            "wall-clock window (tools/cell_admission.py sample), launched through the canonical "
            "detached launcher; purely observational, the cells were never touched. The N=1 "
            "baseline is SECOND-HAND (measured by MAIN 2026-09-04, window length unrecorded)."
        ),
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=CONTENTION_LEDGER,
            reactivation_criteria=(
                "Recalibrate on ANY new contention row: a first-party N=1 baseline measured with a "
                "recorded window, an N=3 row, or a row on different cell shapes. One concurrent "
                "observation cannot establish a scaling law and must not be extrapolated."
            ),
            measurement_axis="[macOS-Metal wall-clock]",
            hardware_substrate="darwin_arm64_apple_silicon_m5_max_128gib",
            captured_at_utc="2026-09-04T15:55:37Z",
        ),
        empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
        noise_floor=None,
        noise_floor_provenance=(
            "NOT MEASURED: a single 420 s window yields no variance estimate. Any comparison of "
            "two speedups closer than the unmeasured floor is unresolvable."
        ),
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="Metal concurrency speedup for local training cells (N=2 anchor)",
        one_line_summary=(
            "A second concurrent Metal cell raised throughput to 31.2854 vs 28.0 steps/min serial "
            "(speedup 1.1173, per-cell efficiency 0.5586); ONE N=2 anchor, not a scaling law."
        ),
        latex_form=(
            r"S(N) = \frac{\sum_{i=1}^{N} r_i}{r_{\mathrm{serial}}}, \quad "
            r"\eta(N) = \frac{S(N)}{N}; \quad "
            r"S(2) = \frac{31.2854}{28.0} = 1.1173, \; \eta(2) = 0.5586"
        ),
        python_callable_module_path=(
            "tools.register_metal_concurrency_speedup_equation:concurrency_speedup"
        ),
        domain_of_validity={
            "hardware_substrate": "darwin_arm64_apple_silicon_m5_max_128gib",
            "backend": "MLX / Apple Metal",
            "measurement_axis": "[macOS-Metal wall-clock]",
            "workload": "QBR1 born-fairform burn-prep cells, 600-pair population, chunk_pairs 16",
            "concurrency_anchored": [1, 2],
            "concurrency_extrapolation_permitted": False,
            "anchors_total": 1,
            "serial_baseline_provenance": "SECOND_HAND (MAIN 2026-09-04; window unrecorded)",
            "noise_floor_measured": False,
            "promotion_authority": False,
            "score_claim": False,
            "scope_note": (
                "Wall-clock throughput is NOT a contest score and never promotes an archive. "
                "The single N=2 anchor licenses exactly one decision -- a second cell paid on "
                "this box, for these cell shapes, on this day. N=3 is UNMEASURED: per-cell "
                "efficiency is already 0.5586 at N=2, so the marginal third cell may be net "
                "negative and nothing here predicts it."
            ),
        },
        units_in={
            "total_steps_per_min": "training_steps_per_minute_summed_over_live_cells",
            "serial_baseline_steps_per_min": "training_steps_per_minute_single_cell",
            "n_cells": "count_of_concurrent_cells",
        },
        units_out={
            "concurrency_speedup": "dimensionless_ratio_ge_1_means_concurrency_paid",
            "per_cell_efficiency": "dimensionless_fraction_of_solo_rate_retained",
        },
        empirical_anchors=(anchor,),
        # Residual of the NULL ("concurrency buys nothing") against the measurement, per axis.
        # Magnitude only: the registry requires non-negative residuals; the SIGN (concurrency
        # paid, it did not cost) is carried by the anchor's signed ``residual`` and by
        # ``one_line_summary``.
        predicted_vs_empirical_residual={
            "total_steps_per_min_vs_serial_null": abs(
                CONCURRENT_TOTAL_STEPS_PER_MIN_N2 - SERIAL_BASELINE_STEPS_PER_MIN
            ),
        },
        last_calibration_utc=now,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_producers=(
            "tools/cell_admission.py:sample_cell_rates",
            "tools/cell_admission.py:append_contention_row",
        ),
        canonical_consumers=(
            "tools/cell_admission.py:throughput_verdict",
            "tools/cell_queue_driver.py:admission_for",
            "tools/costate_digest.py:section_live_cells",
        ),
        provenance=build_provenance_for_predicted(
            model_id=EQUATION_ID,
            inputs_sha256=_inputs_sha256(anchor_inputs),
            measurement_axis="[macOS-Metal wall-clock]",
            hardware_substrate="darwin_arm64_apple_silicon_m5_max_128gib",
            captured_at_utc=now,
        ),
    )


def main() -> int:
    existing = get_equation_by_id(EQUATION_ID)
    if existing is not None:
        print(json.dumps({"status": "ALREADY_REGISTERED", "equation_id": EQUATION_ID}, indent=2))
        return 0
    equation = register_canonical_equation(
        build_equation(),
        agent="ddm_gv1",
        notes=(
            "EQUATIONS-LEG for ddm_gv1: the Metal-contention law the governor gates admission on. "
            f"One measured N=2 anchor; memo {MEMO}."
        ),
    )
    print(
        json.dumps(
            {
                "status": "REGISTERED",
                "equation_id": equation.equation_id,
                "speedup_n2": concurrency_speedup(
                    CONCURRENT_TOTAL_STEPS_PER_MIN_N2, SERIAL_BASELINE_STEPS_PER_MIN
                ),
                "per_cell_efficiency_n2": per_cell_efficiency(
                    CONCURRENT_TOTAL_STEPS_PER_MIN_N2, SERIAL_BASELINE_STEPS_PER_MIN, 2
                ),
                "anchors": len(equation.empirical_anchors),
                "score_claim": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
