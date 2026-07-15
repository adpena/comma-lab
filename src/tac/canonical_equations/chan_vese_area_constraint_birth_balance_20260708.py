# SPDX-License-Identifier: MIT
"""Canonical equation: CHAN-VESE AREA-CONSTRAINT BIRTH-BALANCE law — the level-set precision
counter-force that un-floors the majority (Road) class in the rare-class-birth arm (crucible v7.5).

## The measured problem (road_anomaly_probe_20260708.md, crucible_v6 run-1, ep125, n600)

The rare-class-birth loss stack (seed-islands + island-amplify + persistence-recall + logit-adjust)
applies RECALL/GROWTH pressure on the born-empty classes {Lane=1, Movable=3} with NO matching
PRECISION/area cap. Measured over-paint (witness partition area vs GT):

    class     part_frac   GT area    ratio
    Road0     0.1407      0.2323     0.61x  (UNDER)
    Lane1     0.0805      0.00585    13.8x  OVER
    Movable3  0.0568      0.0124      4.6x  OVER

Mass conservation is exact: the majority DEFICIT (Road 0.0916 + Undriv 0.0273 = 0.1189) equals the
rare EXCESS (Lane 0.0747 + Movable 0.0444 = 0.1191). The over-painted lane+movable area is stolen
almost entirely from Road: ~9% of GT-Road pixels are assigned to lane/movable, and every one is a
Road flip => Road d_seg pinned at a ~0.40 floor, independent of capacity or training time.

## The level-set derivation (operator directive 2026-07-08: "level set and Morse-Smale are perfect
## for engineering the precisely desired annealing behavior")

Formulate the counter-force as the AREA-CONSTRAINT Lagrange term of the level-set (Chan-Vese) region
energy, NOT an ad-hoc penalty. For each birthed class ``c`` with level-set field ``phi_c`` and region
``Omega_c = {phi_c > 0}`` of area ``A_c(phi) = INT H(phi_c)``, add the ONE-SIDED region energy

    E_area,c(phi) = (lambda_c / 2) * max(0, A_c(phi) - A_c^GT)^2          (one-sided quadratic)

whose variational gradient flow is boundary-localized via the Dirac delta ``delta(phi_c)``:

    dphi_c/dt |_area = -dE/dphi_c = -lambda_c * max(0, A_c - A_c^GT) * delta(phi_c)

i.e. an INWARD RETRACTION FLOW on the boundary, magnitude PROPORTIONAL to the area overshoot. Compose
it with the birth force ``F_birth,c`` (the recall/amplify/logit-adjust growth pressure, also boundary-
localized): the region evolves under

    dphi_c/dt = [ F_birth,c - lambda_c * max(0, A_c - A_c^GT) ] * delta(phi_c).

**Equilibrium (the spec — no ramp schedule needed):** growth and constraint BALANCE where the bracket
vanishes,

    F_birth,c = lambda_c * (A_c^* - A_c^GT)   =>   A_c^* = A_c^GT + F_birth,c / lambda_c.

The equilibrium is a STABLE attractor (the quadratic well restores any excursion), so the desired
area is engineered by CHOOSING lambda_c, not by scheduling a decay. The one-sidedness leaves the birth
force UNOPPOSED below A_c^GT (the class still nucleates freely from empty), so this is a precision cap,
not an annihilator.

**Discrete / differentiable form (consume the telemetry, do not rebuild it):** ``A_c(phi)`` is the
differentiable soft-argmax partition mass ``m_c = mean_px softmax(s/T)_c`` of the REALIZED through-R
SegNet logits ``s`` (the SAME forward the island levers already compute — zero extra forward), and
``A_c^GT = mean_px [lstar == c]`` (the GT area fraction, from the cached L*). The loss is

    L_area = SUM_{c in birthed} (lambda_c / 2) * relu(m_c - A_c^GT)^2,

and ``dL/ds ∝ lambda_c * relu(m_c - A_c^GT) * softmax_c*(1 - softmax_c)`` is boundary-localized (the
softmax Jacobian peaks at the codim-1 annulus where the argmax is uncertain) — the discrete
``delta(phi_c)``. So the FORCE is linear in the overshoot (matches "retraction proportional to
overshoot"), the POTENTIAL is the quadratic (matches the equilibrium formula above).

## Deriving lambda_c from the measured ep125 imbalance (the balance arithmetic)

Choose ``lambda_c`` so the equilibrium overshoot is a target fraction ``delta`` of the GT area
(``A_c^* - A_c^GT = delta * A_c^GT``, i.e. equilibrium ratio ``A_c^*/A_c^GT = 1 + delta`` — the SAME
relative overshoot for every birthed class, an elegant single knob):

    lambda_c = F_birth,c / (delta * A_c^GT).

``F_birth,c`` is the birth force in loss-gradient units; its config proxy is the birth loss WEIGHT
``W_birth = amplify_weight = persistence_recall_weight = 1.0`` (the v7 argv value — a MEASURED-ANCHOR
config-conditional constant; re-derive if either weight changes). With ``delta = 0.25`` (equilibrium
at 1.25x GT — loose enough not to strangle the nucleation, tight enough to return the stolen Road
area) and the measured n600 GT areas:

    lambda_lane    = 1.0 / (0.25 * 0.00585) = 683.8
    lambda_movable = 1.0 / (0.25 * 0.0124 ) = 322.6

**Dominance check at the measured ep125 runaway** (the operator's "at 13.8x overshoot the retraction
MUST dominate the birth force"): the retraction force there is ``lambda_c * (A_c(ep125) - A_c^GT)``
= ``W_birth * (r_c^obs - 1) / delta`` where ``r_c^obs`` is the measured ratio:

    lane    : W_birth * (13.76 - 1) / 0.25 = 51.0 * W_birth   => retraction DOMINATES 51x
    movable : W_birth * ( 4.58 - 1) / 0.25 = 14.3 * W_birth   => retraction DOMINATES 14x

so the runaway is retracted hard, and the flow settles at ``1 + delta`` = 1.25x GT.

**Area returned to Road at equilibrium** (why this un-floors Road):

    lane    : 0.0805 -> 1.25 * 0.00585 = 0.00731  (returns 0.0732)
    movable : 0.0568 -> 1.25 * 0.0124  = 0.01550  (returns 0.0413)
    total returned to the majority classes ~ 0.1145 = ~96% of the 0.1189 measured Road+Undriv deficit

so at delta=0.25 the 1.25x GT equilibrium returns ~96% of the stolen area — the Road d_seg floor is
SUBSTANTIALLY lifted but NOT fully un-floored (0.1145 < 0.1189; the last ~4% ~= 0.0044 area remains
over-painted at the 1.25x cap). Closing that residual rides either a tighter tolerance delta (a lower
cap returns more area) or the owed lambda-SCALE A/B; the FORM + dominance are scale-robust either way.

## Epistemic status (NO-FAKE honest tiers)

* The FORM (Chan-Vese one-sided area constraint), the BALANCE law (A* = A_GT + F_birth/lambda), the
  boundary-localization (delta(phi) via the softmax Jacobian), and the balance ARITHMETIC above are
  DERIVED / VERIFIED from the measured ep125 imbalance.
* ``F_birth = W_birth = 1.0`` is a MEASURED-ANCHOR config-conditional proxy (the birth loss weight);
  the ABSOLUTE lambda SCALE that lands the best d_seg is ASSUMED_AWAITING_VERIFICATION until the v7.5
  A/B arm measures it (sister of DirectionalBasisRebalance's sqrt-optimum "owed to the A/B").
* This is a MEANS. Only a byte-closed n600 exact row < 0.19110 from upstream/evaluate.py moves the
  pointer (0.19110 UNMOVED).

DSL leg: ``tac.witness_dsl.curriculum_dsl.AreaConstraintBirth`` (emits ``--area-constraint-birth`` +
knobs). Trainer leg: ``experiments/train_levelset_witness_realized_through_R_mlx.py`` (the L_area term
delegates lambda_c to :func:`area_constraint_lambda`). Completion-event sister:
``tac.witness_control.birth_completion`` (the Morse-Smale persistence hand-off; defense-in-depth).
"""
from __future__ import annotations

import numpy as np

from tac.canonical_equations.equation import (
    ASSUMED_AWAITING_VERIFICATION,
    RECALIBRATE_ON_NEW_ANCHORS,
    VERIFIED_VIA_EMPIRICAL_ANCHOR,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import (
    build_provenance_for_predicted,
    build_provenance_for_research_sidecar,
)

EQUATION_ID = "chan_vese_area_constraint_birth_balance_v1"

_UTC = "2026-07-08T00:00:00Z"
_ADVISORY = "[macOS-MLX research-signal]"
_PREDICTED = "[predicted]"
_PROBE = ".omx/research/road_anomaly_probe_20260708.md"
_MEMO = ".omx/research/v75_birth_counterforce_20260708.md"

# --- MEASURED constants (road_anomaly_probe_20260708.md, crucible_v6 run-1 ep125, n600) -----------
# canonical comma10k class order [Road0, Lane1, Undrivable2, Movable3, MyCar4] (NEVER the luma-sort).
MEASURED_GT_AREA_N600 = (0.2323, 0.00585, 0.4952, 0.0124, 0.2541)
#: witness partition area (part_frac) at ep125 — the runaway state the counter-force must retract.
MEASURED_PART_FRAC_EP125 = (0.1407, 0.0805, 0.4679, 0.0568, 0.2541)
#: the two born-empty rare classes the birth stack targets (Lane, Movable).
BIRTHED_CLASSES = (1, 3)

#: birth-force proxy = the birth loss weight (amplify_weight = persistence_recall_weight in v7 argv).
#: MEASURED-ANCHOR config-conditional; re-derive if either weight changes.
DEFAULT_BIRTH_FORCE = 1.0
#: equilibrium overshoot tolerance: equilibrium area = (1 + delta) * A_GT. DERIVED design choice
#: (loose enough to keep nucleation, tight enough to return the stolen Road area; see module docstring).
DEFAULT_AREA_TOLERANCE = 0.25
#: floor on A_GT so a (near-)absent birthed class never yields lambda = birth_force/0 = +inf.
AREA_FLOOR = 1e-6


def area_constraint_lambda(
    gt_area: float, *, birth_force: float = DEFAULT_BIRTH_FORCE,
    tolerance: float = DEFAULT_AREA_TOLERANCE, floor: float = AREA_FLOOR,
) -> float:
    """The per-class Chan-Vese area-constraint stiffness ``lambda_c = birth_force / (tolerance *
    max(A_c^GT, floor))`` (see the module docstring's balance derivation).

    Chosen so the one-sided quadratic area energy reaches equilibrium at ``A_c^* = (1 + tolerance) *
    A_c^GT``. ``gt_area`` = the class GT area fraction (in [0, 1]); ``birth_force`` = the birth loss
    weight proxy; ``tolerance`` = the equilibrium overshoot fraction. Pure / fail-closed."""
    a = float(gt_area)
    if not np.isfinite(a) or a < 0.0:
        raise ValueError(f"gt_area must be finite and >= 0, got {gt_area!r}")
    if not (float(birth_force) >= 0.0) or not np.isfinite(float(birth_force)):
        raise ValueError(f"birth_force must be finite and >= 0, got {birth_force!r}")
    if not (float(tolerance) > 0.0) or not np.isfinite(float(tolerance)):
        raise ValueError(f"tolerance must be finite and > 0, got {tolerance!r}")
    if not (float(floor) > 0.0):
        raise ValueError(f"floor must be > 0, got {floor!r}")
    return float(birth_force) / (float(tolerance) * max(a, float(floor)))


def equilibrium_overshoot(lambda_c: float, *, birth_force: float = DEFAULT_BIRTH_FORCE) -> float:
    """The equilibrium area overshoot ``A_c^* - A_c^GT = birth_force / lambda_c`` (the balance law).
    Pure; the inverse of :func:`area_constraint_lambda` up to the ``tolerance * A_GT`` grouping."""
    lc = float(lambda_c)
    if not (lc > 0.0) or not np.isfinite(lc):
        raise ValueError(f"lambda_c must be finite and > 0, got {lambda_c!r}")
    return float(birth_force) / lc


def area_penalty(
    soft_mass: np.ndarray, gt_area: np.ndarray, birthed_classes, *,
    birth_force: float = DEFAULT_BIRTH_FORCE, tolerance: float = DEFAULT_AREA_TOLERANCE,
    floor: float = AREA_FLOOR,
) -> float:
    """NUMPY REFERENCE of the trainer's differentiable area-constraint loss (one math, two backends —
    the MLX twin lives in the trainer's ``total_loss_fn``).

    ``L_area = SUM_{c in birthed} (lambda_c / 2) * relu(m_c - A_c^GT)^2`` with ``lambda_c`` from
    :func:`area_constraint_lambda`. ``soft_mass`` / ``gt_area``: ``(K,)`` per-class arrays (the realized
    soft partition mass ``m_c`` and the GT area fraction ``A_c^GT``). One-sided (relu) => zero below
    GT area (nucleation unopposed). Pure / unit-tested."""
    m = np.asarray(soft_mass, np.float64).reshape(-1)
    g = np.asarray(gt_area, np.float64).reshape(-1)
    if m.shape != g.shape:
        raise ValueError(f"soft_mass {m.shape} and gt_area {g.shape} must match")
    total = 0.0
    for c in birthed_classes:
        ci = int(c)
        if ci < 0 or ci >= m.shape[0]:
            raise ValueError(f"birthed class {ci} out of range for K={m.shape[0]}")
        lam = area_constraint_lambda(g[ci], birth_force=birth_force, tolerance=tolerance, floor=floor)
        over = max(0.0, float(m[ci]) - float(g[ci]))
        total += 0.5 * lam * over * over
    return float(total)


def dominance_at_runaway(
    part_frac: float, gt_area: float, *, tolerance: float = DEFAULT_AREA_TOLERANCE,
) -> float:
    """The retraction-force / birth-force RATIO at a measured runaway state:
    ``lambda_c * (A - A_GT) / F_birth = (A/A_GT - 1) / tolerance``.

    > 1 => the area-constraint retraction dominates the birth force there (retracts). At the measured
    ep125 lane ratio 13.76 with tolerance 0.25 this returns ~51 (the operator's dominance requirement).
    Pure; independent of the birth_force magnitude (it cancels)."""
    g = float(gt_area)
    if not (g > 0.0):
        raise ValueError(f"gt_area must be > 0, got {gt_area!r}")
    if not (float(tolerance) > 0.0):
        raise ValueError(f"tolerance must be > 0, got {tolerance!r}")
    return (float(part_frac) / g - 1.0) / float(tolerance)


def build_chan_vese_area_constraint_birth_balance_v1() -> CanonicalEquation:
    """Build the Chan-Vese area-constraint birth-balance law (Lever-1 of crucible v7.5)."""
    _lam_lane = area_constraint_lambda(MEASURED_GT_AREA_N600[1])
    _lam_mov = area_constraint_lambda(MEASURED_GT_AREA_N600[3])

    anchor_imbalance = EmpiricalAnchor(
        anchor_id="rareclass_overpaint_road_floor_ep125_probe_20260708",
        measurement_utc=_UTC,
        inputs={
            "source": "road_anomaly_probe_20260708.md (crucible_v6 run-1, ep125, n600, handoff_readiness "
                      "part_frac vs GT area; [macOS-CPU advisory] NON-PROMOTABLE)",
            "class_order": "[Road0, Lane1, Undrivable2, Movable3, MyCar4] canonical comma10k",
        },
        predicted_output={
            "law": "rare-class birth with recall/growth pressure and NO precision/area cap imposes a "
                   "majority-class d_seg FLOOR = the conserved over-painted area (capacity/time-"
                   "independent)",
        },
        empirical_output={
            "part_frac_ep125": list(MEASURED_PART_FRAC_EP125),
            "gt_area": list(MEASURED_GT_AREA_N600),
            "lane_ratio": round(MEASURED_PART_FRAC_EP125[1] / MEASURED_GT_AREA_N600[1], 3),
            "movable_ratio": round(MEASURED_PART_FRAC_EP125[3] / MEASURED_GT_AREA_N600[3], 3),
            "road_floor_dseg": 0.398,
            "mass_conservation": "majority deficit 0.1189 ~= rare excess 0.1191 (exact)",
        },
        residual=0.0,
        source_artifact=_PROBE,
        measurement_method="existing-telemetry part_frac vs GT bincount, n600 (theta-independent read)",
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_PROBE,
            reactivation_criteria="re-measure on the v7.5 arm (the counter-force config); part_frac is "
                                  "config-conditional",
            measurement_axis=_ADVISORY,
            hardware_substrate="apple_m5_max_cpu_mlx",
        ),
    )
    anchor_lambda_scale_owed = EmpiricalAnchor(
        anchor_id="chan_vese_area_constraint_lambda_scale_owed_v75_ab_20260708",
        measurement_utc=_UTC,
        inputs={
            "law": "lambda_c = birth_force / (tolerance * A_GT_c); equilibrium A* = A_GT + F_birth/lambda",
            "birth_force_proxy": "W_birth = amplify_weight = persistence_recall_weight = 1.0 (v7 argv, "
                                 "config-conditional)",
            "tolerance": DEFAULT_AREA_TOLERANCE,
            "derived_lambda": {"lane": round(_lam_lane, 1), "movable": round(_lam_mov, 1)},
            "dominance_at_ep125": {
                "lane": round(dominance_at_runaway(MEASURED_PART_FRAC_EP125[1],
                                                   MEASURED_GT_AREA_N600[1]), 1),
                "movable": round(dominance_at_runaway(MEASURED_PART_FRAC_EP125[3],
                                                      MEASURED_GT_AREA_N600[3]), 1),
            },
        },
        predicted_output={
            "claim": "equilibrium at 1.25x GT returns ~0.11 area to the majority (>= the 0.1189 measured "
                     "Road+Undriv deficit) => Road d_seg floor lifts",
        },
        empirical_output={"status": "OWED — the v7.5 area-constraint A/B arm measures the best lambda scale"},
        residual=0.0,
        source_artifact=_MEMO,
        measurement_method="derivation only (no counter-force run yet; v7.5 A/B owed)",
        empirical_verification_status=ASSUMED_AWAITING_VERIFICATION,
        provenance=build_provenance_for_predicted(
            model_id="chan_vese_area_constraint.lambda_scale",
            inputs_sha256="0" * 64,
            measurement_axis=_PREDICTED,
            hardware_substrate="apple_m5_max_cpu_mlx",
        ),
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name=("Chan-Vese area-constraint birth-balance: one-sided quadratic level-set region energy "
              "E=(lambda_c/2)relu(A_c-A_c^GT)^2 caps rare-class over-paint; lambda_c=W_birth/(delta*A_GT) "
              "balances birth at A*=(1+delta)A_GT (un-floors the majority Road class)"),
        one_line_summary=(
            "one-sided Chan-Vese area constraint on the birthed classes' realized soft mass; "
            "lambda_c=W_birth/(delta*A_GT_c) balances birth at (1+delta)*A_GT, returning over-paint to Road."
        ),
        latex_form=(
            r"E_{\mathrm{area},c}=\tfrac{\lambda_c}{2}\max(0,A_c-A_c^{GT})^2,\quad"
            r"\lambda_c=\frac{F_{\mathrm{birth}}}{\delta\,A_c^{GT}},\quad"
            r"A_c^{*}=A_c^{GT}+\frac{F_{\mathrm{birth}}}{\lambda_c}=(1+\delta)A_c^{GT}"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.chan_vese_area_constraint_birth_balance_20260708:"
            "area_constraint_lambda"
        ),
        domain_of_validity={
            "vehicle": ["softmax_of_sdf_levelset_witness"],
            "lever": ("tac.witness_dsl.curriculum_dsl.AreaConstraintBirth (--area-constraint-birth + "
                      "--area-constraint-classes/-birth-force/-tolerance); consumes the realized soft "
                      "partition mass the island levers already compute (no extra forward)"),
            "measurement_axis": ["macOS-MLX research-signal", "predicted"],
            "note": ("lambda_c is a GLOBAL STIFFNESS: the trainer derives it LIVE from the GLOBAL GT "
                     "area (bincount over the whole L* stack), deliberately — a per-pair lambda would "
                     "blow up (lambda->inf) on a pair where the class is absent (A_GT_c=0). The "
                     "PENALTY TARGET A_GT_c is PER-PAIR (mean_px lstar_oh_c). So the per-pair "
                     "equilibrium ratio is 1 + delta*(A_global/A_pair), exactly 1+delta only at the "
                     "average pair (SOUND / safer by design — a global-stiffness spring, a per-pair "
                     "target). The n600 constants here are the measured global anchor. The lambda "
                     "SCALE is owed to the v7.5 A/B; the FORM + balance are derived."),
        },
        units_in={"gt_area": "fraction_of_pixels", "birth_force": "loss_weight", "tolerance": "dimensionless"},
        units_out={"lambda_c": "loss_weight_per_area^2"},
        empirical_anchors=(anchor_imbalance, anchor_lambda_scale_owed),
        predicted_vs_empirical_residual={
            "equilibrium_ratio": 1.0 + DEFAULT_AREA_TOLERANCE,
            "lane_dominance_at_ep125": round(
                dominance_at_runaway(MEASURED_PART_FRAC_EP125[1], MEASURED_GT_AREA_N600[1]), 1),
        },
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "experiments/train_levelset_witness_realized_through_R_mlx.py",
            "tac.witness_dsl.curriculum_dsl",
            "tac.witness_autoconfig",
        ),
        canonical_producers=(
            "experiments/train_levelset_witness_realized_through_R_mlx.py",
        ),
        provenance=build_provenance_for_predicted(
            model_id="chan_vese_area_constraint_birth_balance.v1",
            inputs_sha256="0" * 64,
            measurement_axis=_PREDICTED,
            hardware_substrate="apple_m5_max_cpu_mlx",
        ),
    )


def populate_chan_vese_area_constraint_birth_balance_equation(
    *, path=None, lock_path=None, agent: str | None = None, subagent_id: str | None = None,
) -> CanonicalEquation:
    """Idempotent APPEND-ONLY registration of the Chan-Vese area-constraint birth-balance law.
    Equations leg of crucible v7.5 Lever-1; DSL leg = AreaConstraintBirth; trainer leg =
    --area-constraint-birth."""
    from tac.canonical_equations.registry import register_canonical_equation

    eq = build_chan_vese_area_constraint_birth_balance_v1()
    register_canonical_equation(
        eq, path=path, lock_path=lock_path, agent=agent, subagent_id=subagent_id,
        notes="chan_vese_area_constraint_birth_balance_20260708 (equations leg of crucible v7.5 "
              "birth-counter-force Lever-1; DSL leg = AreaConstraintBirth; trainer leg = "
              "--area-constraint-birth; lambda SCALE = owed to the v7.5 A/B anchor)",
    )
    return eq


# --- Distinct append-only law: isoperimetric birth-weight scaling (2026-07-15) -----------------
# This law is intentionally separate from the area-retraction law above.  The existing 51x/14.3x
# runaway facts describe a precision counter-force; this law describes the DERIVED relative birth
# support needed to oppose each class's perimeter/area homogenization drain.  Its absolute scale and
# efficacy remain unmeasured.
ISOPERIMETRIC_BIRTH_WEIGHT_EQUATION_ID = "isoperimetric_birth_weight_scaling_v1"
ISLAND_BIRTH_RATIO_RECEIPT_PATH = (
    "src/tac/canonical_equations/island_birth_isoperimetric_ratio_n96.json"
)
ISLAND_BIRTH_RATIO_RECEIPT_SHA256 = (
    "74129e133869ddaff1f8b4b3198fde4b45ceaffcc5aaa52ef1f18aab59f62ef7"
)
ISLAND_BIRTH_MEASUREMENT_MEMO = (
    ".omx/research/island_birth_saddle_node_hysteresis_measurement_20260715.md"
)
ISLAND_BIRTH_DAG_FEED = (
    ".omx/research/sub015_DAG_exp_linear_reparam_warmstart_20260714.md:105-111"
)


def isoperimetric_birth_weight(
    absolute_scale: float,
    class_p_over_a: float,
    reference_p_over_a: float,
) -> float:
    """Return ``absolute_scale * class_p_over_a / reference_p_over_a``.

    The ratio is the DERIVED class geometry law ``W_birth,c proportional to (P/A)_c``.  The
    ``absolute_scale`` remains an explicitly tunable, UNMEASURED radius scale.  All inputs must be
    finite and strictly positive so a corrupt receipt or invalid authoring value fails closed.
    """
    scale = float(absolute_scale)
    class_ratio = float(class_p_over_a)
    reference_ratio = float(reference_p_over_a)
    for name, value in (
        ("absolute_scale", scale),
        ("class_p_over_a", class_ratio),
        ("reference_p_over_a", reference_ratio),
    ):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and > 0, got {value!r}")
    return scale * class_ratio / reference_ratio


def build_isoperimetric_birth_weight_scaling_v1() -> CanonicalEquation:
    """Build the relative island-birth support law from the sealed n96 geometry receipt."""
    geometry_anchor = EmpiricalAnchor(
        anchor_id="island_birth_isoperimetric_ratio_n96_20260715",
        measurement_utc="2026-07-15T00:00:00Z",
        inputs={
            "receipt": ISLAND_BIRTH_RATIO_RECEIPT_PATH,
            "receipt_sha256": ISLAND_BIRTH_RATIO_RECEIPT_SHA256,
            "source": "experiments/results/mlx_fleet_gt_cache/gt_n96.npz",
            "source_sha256": "6aad6600d93a5c25e94207ee411d3b4daf93136b8ea4235b6f7b9d96f04ab104",
            "extraction": (
                "lstars; canonical Lane=1 and Movable=3; boundary pixels are class pixels not "
                "surviving a 4-connected cross erosion with constant-false border; aggregate n96"
            ),
        },
        predicted_output={
            "law": "W_birth,c proportional to (P/A)_c",
            "absolute_scale": "UNMEASURED; remains a positive tunable DSL authoring value",
        },
        empirical_output={
            "lane": {"area": 111081, "boundary": 84389, "p_over_a": 0.759706880564633},
            "movable": {"area": 293859, "boundary": 25137, "p_over_a": 0.08554102477718906},
            "lane_over_movable_ratio": 8.881199197033954,
            "efficacy": "UNMEASURED — ratio is not evidence that firing improves d_seg or score",
        },
        residual=0.0,
        source_artifact=ISLAND_BIRTH_RATIO_RECEIPT_PATH,
        measurement_method=(
            "frozen n96 GT geometry extraction; exact counts and source custody are sealed in the "
            "content-addressed receipt"
        ),
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=ISLAND_BIRTH_RATIO_RECEIPT_PATH,
            reactivation_criteria=(
                "only an operator-authorized live resumed W_birth up/down ramp may establish "
                "efficacy or an absolute threshold"
            ),
            measurement_axis=_ADVISORY,
            hardware_substrate="apple_m5_max_cpu",
        ),
    )
    return CanonicalEquation(
        equation_id=ISOPERIMETRIC_BIRTH_WEIGHT_EQUATION_ID,
        name=(
            "Isoperimetric island-birth support scaling: class birth weight is proportional to "
            "the class perimeter/area homogenization drain"
        ),
        one_line_summary=(
            "W_birth,c = W_ref * (P/A)_c / (P/A)_ref; n96 geometry fixes only the relative "
            "Lane/Movable ratio (8.881199...), while the absolute scale and efficacy remain owed."
        ),
        latex_form=(
            r"W_{\mathrm{birth},c}=W_{\mathrm{ref}}"
            r"\frac{(P/A)_c}{(P/A)_{\mathrm{ref}}},\qquad"
            r"\frac{W_{\mathrm{Lane}}}{W_{\mathrm{Movable}}}=8.881199197\ldots"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.chan_vese_area_constraint_birth_balance_20260708:"
            "isoperimetric_birth_weight"
        ),
        domain_of_validity={
            "vehicle": ["softmax_of_sdf_levelset_witness"],
            "lever": "tac.witness_dsl.curriculum_dsl.LadderIslandHomotopy",
            "reference_class": "Lane=1",
            "measurement_axis": ["macOS-CPU advisory", "derived"],
            "evidence_scope": (
                "relative per-class geometry only; saddle-node/hysteresis NOT CONFIRMED; no efficacy "
                "or score claim; absolute scale requires a live resumed up/down ramp"
            ),
            "triality": {
                "dag": ISLAND_BIRTH_DAG_FEED,
                "dsl": "tac.witness_dsl.curriculum_dsl.LadderIslandHomotopy",
                "equation": ISOPERIMETRIC_BIRTH_WEIGHT_EQUATION_ID,
            },
        },
        units_in={
            "absolute_scale": "pixels",
            "class_p_over_a": "inverse_pixels",
            "reference_p_over_a": "inverse_pixels",
        },
        units_out={"class_birth_radius": "pixels"},
        empirical_anchors=(geometry_anchor,),
        predicted_vs_empirical_residual={
            "relative_ratio_residual": 0.0,
        },
        last_calibration_utc="2026-07-15T00:00:00Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.witness_dsl.curriculum_dsl.LadderIslandHomotopy",
            "tac.witness_dsl.lawref",
        ),
        canonical_producers=(
            ISLAND_BIRTH_MEASUREMENT_MEMO,
            ISLAND_BIRTH_RATIO_RECEIPT_PATH,
        ),
        provenance=build_provenance_for_predicted(
            model_id="isoperimetric_birth_weight_scaling.v1",
            inputs_sha256=ISLAND_BIRTH_RATIO_RECEIPT_SHA256,
            measurement_axis=_PREDICTED,
            hardware_substrate="apple_m5_max_cpu",
        ),
    )


def populate_isoperimetric_birth_weight_scaling_equation(
    *, path=None, lock_path=None, agent: str | None = None, subagent_id: str | None = None,
) -> CanonicalEquation:
    """Idempotently append the isoperimetric island-birth ratio law to a registry."""
    from tac.canonical_equations.registry import register_canonical_equation

    eq = build_isoperimetric_birth_weight_scaling_v1()
    register_canonical_equation(
        eq,
        path=path,
        lock_path=lock_path,
        agent=agent,
        subagent_id=subagent_id,
        notes=(
            "isoperimetric island-birth ratio law; Lane/Movable=8.881199... DERIVED from sealed "
            "n96 GT geometry; absolute scale and efficacy remain unmeasured"
        ),
    )
    return eq


__all__ = [
    "AREA_FLOOR",
    "BIRTHED_CLASSES",
    "DEFAULT_AREA_TOLERANCE",
    "DEFAULT_BIRTH_FORCE",
    "EQUATION_ID",
    "ISLAND_BIRTH_DAG_FEED",
    "ISLAND_BIRTH_MEASUREMENT_MEMO",
    "ISLAND_BIRTH_RATIO_RECEIPT_PATH",
    "ISLAND_BIRTH_RATIO_RECEIPT_SHA256",
    "ISOPERIMETRIC_BIRTH_WEIGHT_EQUATION_ID",
    "MEASURED_GT_AREA_N600",
    "MEASURED_PART_FRAC_EP125",
    "area_constraint_lambda",
    "area_penalty",
    "build_chan_vese_area_constraint_birth_balance_v1",
    "build_isoperimetric_birth_weight_scaling_v1",
    "dominance_at_runaway",
    "equilibrium_overshoot",
    "isoperimetric_birth_weight",
    "populate_chan_vese_area_constraint_birth_balance_equation",
    "populate_isoperimetric_birth_weight_scaling_equation",
]
