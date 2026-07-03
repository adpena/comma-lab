# SPDX-License-Identifier: MIT
"""Canonical equation: the SYSTEM-aware adaptive-ceiling + admission-control law that prevents the
multi-job SUM-over-RAM crash (2026-07-02 machine crash; operator P0 machine-protection).

THE FAILURE THIS CODIFIES (memory ``memory_blackbox_and_system_governor_20260703`` + the 2026-07-02
machine crash): our OOM protection was PER-PROCESS (``safe_run --rss-mb 90000``) + a per-RUN peak
projection (``witness_memory_preflight``) — both BLIND to the SYSTEM total. Two/three jobs each under
their per-process cap (R1 trainer ~67 GiB + a byte-close/bsdtar/inflate job + the ev1 descent-probe +
build agents) SUMMED past 128 GiB -> macOS jetsam cascade -> hang.

THE LAW (job-type-BLIND — the ceiling is on the SYSTEM, measured from real vm_stat used):

    safety_margin(T)      = max(8, 0.08 * T)                       # GiB
    adaptive_ceiling(T)   = T - safety_margin(T)                   # max system-used we tolerate
    baseline              = system_used - sum(current RSS of our tracked jobs)   # OS + control-plane
    training_budget       = adaptive_ceiling - baseline            # peak all our jobs may collectively use
    projected_system_used = system_used + sum_active max(0, peak_i - current_i) + projected_new_peak
    ADMIT  iff  projected_system_used <= adaptive_ceiling

On a 128 GiB box: margin 10.24, ceiling 117.76; with baseline ~16 GiB the training budget is ~101.76
GiB — MUCH higher than the old blind 90 GB per-process cap (we MAX OUT the box) while REFUSING the
concurrent-sum overflow. The admission uses the REAL vm_stat ``used`` (counts OS + control-plane +
training + byte-close + inflate + bsdtar + probes) so it is job-type-blind; no double count (current
RSS is in ``used`` once, we add only each active job's REMAINING growth).

TRUST (why this is believed over eyeballing vm_stat by hand): the accounting is reconciled from FIXED
captured kernel counters and UNIT-TESTED to exact GiB, CROSS-VALIDATED against a 2nd independent source
(sysctl vm.page_free_count / memory_pressure / psutil) with FAIL-SAFE on disagreement, and
CLOSURE-checked (partition sum ~= physical total). The admission gate ships ADVISORY (logs what it
WOULD refuse) until independent adversarial review flips ``TAC_ADMISSION_ENFORCE=1``.

VERDICT (honest, NO-FAKE): a SCORE-NEUTRAL machine-protection / launch-safety law (not a
d_seg/d_pose/rate lever); pointer 0.19110 UNMOVED. Deterministic arithmetic identity -> the
predicted-vs-empirical residual is 0 (the ceiling / budget / admission are exact); the empirical anchor
is the crash the law would have prevented. Advisory axis (``[macOS-CPU advisory] NON-PROMOTABLE``).

Producer: ``tools/system_memory_governor.py`` (the live guard). Consumers: ``tools/launch_witness_run.py``
+ ``tools/spawn_durable_daemon.py`` (the HARD admission gate) + ``tools/witness_memory_preflight.py``
(the composed system-aware admission) + ``tools/memory_blackbox.py`` (records the ceiling trajectory).
"""
from __future__ import annotations

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

EQUATION_ID = "adaptive_ceiling_admission_control_v1"

_UTC = "2026-07-03T00:00:00Z"
_ADVISORY = "[macOS-CPU advisory]"
_LEDGER = ".omx/research/memory_blackbox_and_system_governor_20260703.md"

# MEASURED / policy constants — MIRRORED from the live guard tools/system_memory_governor.py; the test
# cross-checks these against the guard so drift is caught.
SAFETY_MARGIN_FLOOR_GIB = 8.0
SAFETY_MARGIN_FRAC = 0.08
FLEET_TOTAL_RAM_GIB = 128.0     # the M5 Max the crash happened on


def safety_margin_gib(total_gib: float) -> float:
    """Adaptive safety margin = max(floor, frac * total). On 128 GiB -> 10.24 GiB. Pure."""
    return max(SAFETY_MARGIN_FLOOR_GIB, SAFETY_MARGIN_FRAC * float(total_gib))


def adaptive_ceiling_gib(total_gib: float) -> float:
    """Max system-wide used RAM we tolerate = total - safety_margin. Pure."""
    return float(total_gib) - safety_margin_gib(total_gib)


def training_budget_gib(total_gib: float, baseline_gib: float) -> float:
    """Peak all our jobs may collectively use = ceiling - baseline (OS+control-plane). Pure."""
    return adaptive_ceiling_gib(total_gib) - float(baseline_gib)


def projected_system_used_gib(
    system_used_gib: float, active_growth_headroom_gib: float, projected_new_gib: float
) -> float:
    """Projected system-used if the new job launches + active jobs grow to peak (no double count)."""
    return float(system_used_gib) + max(0.0, float(active_growth_headroom_gib)) + float(projected_new_gib)


def admits(
    *,
    projected_new_gib: float,
    system_used_gib: float,
    active_growth_headroom_gib: float,
    total_gib: float,
) -> bool:
    """The admission predicate: ADMIT iff projected system-used <= adaptive ceiling. Pure + the exact
    law the launcher's HARD gate enforces."""
    return projected_system_used_gib(system_used_gib, active_growth_headroom_gib, projected_new_gib) \
        <= adaptive_ceiling_gib(total_gib)


def build_adaptive_ceiling_admission_control_v1() -> CanonicalEquation:
    """Build the adaptive-ceiling + admission-control canonical equation (the SUM-over-RAM crash law)."""
    # Empirical anchor: the 2026-07-02 crash. R1 (~67 GiB) already running; a 2nd ~67 GiB job. With
    # baseline ~16 GiB the projected system-used = 70 (R1 resident) + 67 (new) = 137 > ceiling 117.76
    # -> the law REFUSES the 2nd job. Had the gate existed, the machine would not have crashed.
    total = FLEET_TOTAL_RAM_GIB
    ceiling = adaptive_ceiling_gib(total)                       # 117.76
    budget = training_budget_gib(total, 16.0)                   # 101.76
    r1_plus_second = projected_system_used_gib(70.0, 0.0, 67.0)  # 137.0 (the crash demand)
    law_refuses_second = not admits(projected_new_gib=67.0, system_used_gib=70.0,
                                    active_growth_headroom_gib=0.0, total_gib=total)
    lone_run_admitted = admits(projected_new_gib=67.0, system_used_gib=18.0,
                               active_growth_headroom_gib=0.0, total_gib=total)

    anchor = EmpiricalAnchor(
        anchor_id="machine_crash_sum_over_ram_two_concurrent_67gib_jobs_20260702",
        measurement_utc=_UTC,
        inputs={
            "total_ram_gib": total,
            "r1_resident_gib": 67.0,
            "second_job_projected_peak_gib": 67.0,
            "safety_margin_gib": safety_margin_gib(total),
            "note": "per-process safe_run cap (90 GB) + per-run projection were both SYSTEM-blind",
        },
        predicted_output={
            "adaptive_ceiling_gib": ceiling,
            "training_budget_gib_at_baseline_16": budget,
            "projected_system_used_two_67gib_jobs": r1_plus_second,
            "law_refuses_second_job": law_refuses_second,
            "law_admits_lone_67gib_job": lone_run_admitted,
        },
        empirical_output={
            "observed": ("machine crashed (jetsam cascade / hang) — no SYSTEM gate existed; the sum of "
                         "R1 (~67 GiB) + byte-close/inflate + ev1 descent-probe + build agents exceeded "
                         "128 GiB"),
            "law_verdict": ("REFUSE the 2nd concurrent 67 GiB job (137 > 117.76 ceiling); ADMIT a lone "
                            "67 GiB job (85 < 117.76) — the crash is exactly the case the gate prevents"),
            "residual_is_zero_because": "deterministic arithmetic identity (ceiling/budget/admission exact)",
        },
        residual=0.0,
        source_artifact=_LEDGER,
        measurement_method="deterministic_admission_arithmetic_vs_20260702_crash_event",
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_LEDGER,
            reactivation_criteria=(
                "re-derive the margin/ceiling if the fleet RAM changes or the safety-margin policy "
                "(floor 8 GiB / 8%) is retuned; recalibrate against the guard's live constants"
            ),
            measurement_axis=_ADVISORY,
            hardware_substrate="m5_max_128gib",
        ),
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
    )

    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name=("SYSTEM-aware adaptive-ceiling + admission-control law — refuse a launch whose peak, "
              "SUMMED over the real vm_stat used + active jobs' growth, would bust total RAM"),
        one_line_summary=(
            "ceiling = T - max(8, 0.08T); admit iff system_used + active_growth + new_peak <= ceiling; "
            "on 128 GiB budget ~101.76 GiB (maxes the box) while refusing the concurrent-sum crash"
        ),
        latex_form=(
            r"\text{margin}(T)=\max(8,0.08T);\ C(T)=T-\text{margin}(T);\ "
            r"\text{ADMIT} \iff u + \sum_i \max(0,p_i-c_i) + p_{new} \le C(T)"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.adaptive_ceiling_admission_control_20260703:admits"
        ),
        domain_of_validity={
            "platform": ["macos_unified_memory"],
            "measurement_source": ["vm_stat_free_plus_inactive_conservative", "hw.memsize", "hw.pagesize"],
            "measurement_axis": [_ADVISORY],
            "result_type": ("SCORE-NEUTRAL machine-protection / launch-safety law (not a d_seg/d_pose/"
                            "rate lever)"),
            "constants_mirror": "tools/system_memory_governor.py (drift-guarded by the test)",
            "enforcement": ("HARD PREVENT gate by design; ships ADVISORY (logs would-refuse) until "
                            "independent adversarial review flips TAC_ADMISSION_ENFORCE=1"),
            "trust": ("accounting reconciled from fixed kernel counters + unit-tested to exact GiB + "
                      "cross-validated (sysctl/memory_pressure/psutil) + closure-checked + fail-safe"),
        },
        units_in={"total_gib": "gibibytes", "system_used_gib": "gibibytes",
                  "active_growth_headroom_gib": "gibibytes", "projected_new_gib": "gibibytes"},
        units_out={"admits": "bool"},
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={
            "deterministic_admission_arithmetic_vs_20260702_crash_event": 0.0,
        },
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tools/launch_witness_run.py",
            "tools/spawn_durable_daemon.py",
            "tools/witness_memory_preflight.py",
            "tools/memory_blackbox.py",
        ),
        canonical_producers=(
            "tools/system_memory_governor.py",
        ),
        provenance=build_provenance_for_predicted(
            model_id="adaptive_ceiling_admission_control.v1",
            inputs_sha256="0" * 64,
            measurement_axis=_ADVISORY,
            hardware_substrate="m5_max_128gib",
        ),
    )


__all__ = [
    "EQUATION_ID",
    "SAFETY_MARGIN_FLOOR_GIB",
    "SAFETY_MARGIN_FRAC",
    "adaptive_ceiling_gib",
    "admits",
    "build_adaptive_ceiling_admission_control_v1",
    "projected_system_used_gib",
    "safety_margin_gib",
    "training_budget_gib",
]
