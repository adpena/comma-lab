# SPDX-License-Identifier: MIT
"""Canonical equation: the EXCHANGE RATE of the compensated lossy-semantic-edit family.

What the family is
------------------
Quantize the semantic FiLM tensors more coarsely (buying rate), then repair the pose
damage with an **in-compile Schur compensation** solved against the edited object -- the
mechanism qs5 proved and qs4's disaster (+2.396e-04, a compensation carried from a
DIFFERENT object) named. sa3 is the first member re-based onto a live pointer.

Why this law exists
-------------------
"It bought bytes" and "it cost distortion" are both true of every member of this family,
and neither settles anything. What decides a fire is the RATIO: per S unit of rate credit,
how much comes back as pose and seg damage, and how much survives as net? Without the
ratio each rung is re-argued from scratch, and the family's CEILING -- whether it can close
the gap at all -- is never asked.

MEASURED (sa3, contest-CUDA T4 n600, archive
``d2ad58ee28b84388a262bd5c8b11611a163dcc2694ad3c29a1283605a206b992`` @ 179,140 B, against
the sz1 pointer 0.15771357797660338 @ 179,930 B)::

    rate credit   -5.260286e-04 S   (-790 B)                          1.000 of credit
    pose damage   +2.669654e-04 S   (d_pose 6.880e-06 -> 7.330e-06)   0.508
    seg damage    +2.040000e-04 S   (d_seg +2.04e-06 = +241 flips)    0.388
    ------------------------------------------------------------------------
    NET           -5.506320e-05 S                                     0.105 retained

The legs sum to the harness's own full-precision recompute to 17 digits, so the split is
exact at the reported precision, not a reconciliation.

**Quantization custody, because this campaign has been bitten by it.** The row's
``canonical_score_source`` is ``report_8dp_components_plus_exact_archive_bytes`` -- the score is
built from 8-decimal-place components, and the harness publishes its own worst-case bound:
``report_8dp_score_worst_case_abs_error_bound = 3.4205e-06`` (pose ±2.92e-06, seg ±5e-07). The
net is **16.1x that bound**, so the sign is determinate by a stated margin rather than by
assumption. That is the cure for the ``#1032`` genus, where a −4e-06 "result" turned out to be
one pose ULP wearing a verdict's clothes. Always divide the delta by the bound before believing
it.

THE COUNTER-INTUITIVE PART, and it is the useful part
-----------------------------------------------------
Retention **improves** with edit mass. Rate credit and seg damage are linear in mass; pose
damage is not, because ``sqrt(10*d_pose)`` is CONCAVE and this family pays pose in the
*upward* direction, where the marginal ``5/sqrt(10*d_pose)`` is falling. sa1 measured pose
damage linear in mass at 0.91x of linear, which licenses extrapolating ``d_pose`` and then
re-taking the square root -- never extrapolating the pose S-leg itself.

    mass x1 -> net -5.506e-05, retention 10.47%
    mass x2 -> net -1.182e-04, retention 11.24%
    mass x4 -> net -2.660e-04, retention 12.64%

THE PRECONDITION THIS MODEL CARRIES, and it is NOT free
-------------------------------------------------------
The extrapolation above holds ``residual_fraction`` -- the share of uncompensated pose damage
the in-compile Schur compensation FAILS to cancel -- fixed at sa3's value. That is a
precondition, not a law. sa1's 0.91x mass-linearity was measured on the UNCOMPENSATED family,
where residual == damage; on the compensated family the residual is
``(1 - cancellation) * damage``, and **cancellation is re-solved per object**. It is not a
function of mass.

**MEASURED FALSIFICATION (keep01, 2026-08-18, n600 advisory).** The keep01 row is the same
family at 3.0x sa3's mass (-2,369 B vs -790 B). Naive mass-extrapolation predicts ~11.9%
retention; the arm measured **42.2%** -- a 3.3x under-prediction -- because keep01's
compensation cancels 99.9801% of a 4.44x LARGER uncompensated damage than sa3's row, clearing
its own 99.9479% requirement with 2.62x margin on the residual. Better compensation on a
bigger edit, not the concavity alone.

Consequence: ``project_at_mass`` is CONSERVATIVE when a successor re-solves compensation at
least as well as sa3 did, and WRONG IN THE DANGEROUS DIRECTION if a successor's compensation
degrades. Pass ``residual_fraction_vs_unit`` explicitly whenever the successor's cancellation
has been measured. Carrying sa3's implied cancellation onto a different object unexamined is
the same cross-regime constant-transfer that cost qs4 +2.396e-04.

Everywhere else on this campaign, concavity has been the enemy (a 2x worsening costs 1.41x
what a 2x improvement buys, per ``ddm_asym1``). Here it is the friend, for exactly the same
reason and with the sign flipped.

THE CEILING -- CONDITIONAL, and its falsifier has already fired
----------------------------------------------------------------
Solving the same arithmetic for the mass that closes the 0.00765851 gap ALONE gives
**52.1x**, requiring **41,160 B** of rate credit against a **34,243 B** ``semantic_blob``
(1.20x the whole section) -- **at sa3's compensation quality**.

That qualifier is load-bearing, and the pre-registered falsifier ("a higher-mass rung whose
measured retention beats the prediction") FIRED the same day. keep01, at 3.0x mass, measured
42.2% retention against ~11.9% predicted, implying ``residual_fraction_vs_unit == 0.376``
(2.66x better cancellation). At that quality the demand falls to **25,752 B = 0.75x the
section -- nominally feasible.**

So the honest verdict is NOT "contributor, never a route", which is what a ceiling computed
at one fixed compensation quality wrongly asserted. It is:

* the family is **NOT CLOSED as a route**;
* whether it routes turns entirely on whether compensation quality HOLDS at high mass;
* and 32.6x is **10.9x beyond keep01's measured point**, so the feasible verdict is a
  projection resting on one extrapolated quantity -- the same shape as the infeasible
  verdict it overturned. Neither is a finding.

What WOULD settle it: a measured cancellation-vs-mass curve. Two points (sa3 at 1x, keep01
at 3x) do not have one, and they disagree about the quantity the whole ceiling depends on.
Until that curve exists, call ``family_cannot_close_alone`` with a MEASURED
``residual_fraction_vs_unit`` and read ``feasible`` as conditional on it.

``verdict_scope``: the EXCHANGE RATIOS are INSTANCE (one measured member at one operating
point); the SHAPE (rate/seg linear, pose concave, retention rising with mass) is
DERIVED-EXACT from the score function and does not depend on the anchor.
"""

from __future__ import annotations

from decimal import Decimal, getcontext

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

EQUATION_ID = "compensated_semantic_edit_exchange_v1"
AXIS = "[contest-CUDA T4 n600] measured legs; exchange arithmetic DERIVED-EXACT"
SOURCE_MEMO = ".omx/research/ddm_sa3_compensated_edit_rebased_verdict_20260818.md"

RATE_DENOMINATOR_BYTES = 37_545_489
SEG_PIXELS = 600 * 512 * 384

#: The sa3 anchor, verbatim from MODAL_REMOTE_RESULT.json.
ANCHOR_ARCHIVE_SHA256 = "d2ad58ee28b84388a262bd5c8b11611a163dcc2694ad3c29a1283605a206b992"
ANCHOR_ARCHIVE_BYTES = 179_140
ANCHOR_SCORE = 0.15765851477950737
ANCHOR_D_SEG = 0.00029815
ANCHOR_D_POSE = 7.33e-06

#: The sz1 pointer this member was re-based onto.
BASE_ARCHIVE_SHA256 = "debb025f45bb42e3b8131714cf462a9963e449bc65ff5eade9484fde094b037a"
BASE_ARCHIVE_BYTES = 179_930
BASE_SCORE = 0.15771357797660338
BASE_D_SEG = 0.00029611
BASE_D_POSE = 6.880e-06

#: Measured at mass 1.0 (see the module docstring for the derivation of each).
RATE_CREDIT_S = 5.260286e-04
POSE_DAMAGE_S = 2.669654e-04
SEG_DAMAGE_S = 2.040000e-04
NET_S = -5.506320e-05

POSE_DAMAGE_FRACTION_OF_CREDIT = 0.50751
SEG_DAMAGE_FRACTION_OF_CREDIT = 0.38781
RETENTION_FRACTION_AT_UNIT_MASS = 0.10468

#: Mass multiple whose net alone equals the sz1->0.15 gap, and the bytes of rate credit it
#: demands. The semantic section holds 34,243 B, so this is unreachable -- that IS the
#: finding.
GAP_CLOSING_MASS_MULTIPLE = 52.1017
GAP_CLOSING_RATE_CREDIT_BYTES = 41_160.3
SEMANTIC_SECTION_BYTES = 34_243

#: sa1's measured mass-linearity of pose damage: 0.91x of exactly linear. Extrapolate
#: d_pose with this, then take sqrt(10*d_pose). Never extrapolate the pose S-leg.
#: SCOPE: measured on the UNCOMPENSATED family, where residual == damage.
SA1_POSE_MASS_LINEARITY = 0.91

# --- keep01: the measured falsification of naive mass-extrapolation -------------------
# ddm_sa3 keep01_authority/sm3r_keep01, n600 (all 600 pairs, representativeness 0.99998),
# axis [macOS-CPU advisory, frozen CPU-torch PoseNet], score_claim=False.
KEEP01_MASS_MULTIPLE = 2_369 / 790  # 2.9987x sa3's edit mass
KEEP01_CANCELLATION_ACHIEVED = 0.9998012041204907
KEEP01_CANCELLATION_REQUIRED = 0.9994794021137176
KEEP01_MEAN_RESIDUAL_D_POSE = 7.408064299123584e-07
KEEP01_MAX_RESIDUAL_ALLOWED = 1.940020465520849e-06
KEEP01_PROJECTED_NET_S = -6.652669689221182e-04
KEEP01_MEASURED_RETENTION = 0.42174  # net / |rate credit|, vs ~0.119 naively predicted
#: How far naive mass-extrapolation under-predicted keep01. Not a correction factor to
#: apply blindly -- it is the SIZE of the precondition, and the reason to measure
#: cancellation per object rather than inherit it.
KEEP01_NAIVE_UNDERPREDICTION_FACTOR = 3.3


def _dec(value: float | Decimal) -> Decimal:
    return value if isinstance(value, Decimal) else Decimal(repr(value))


def exchange_legs(
    *,
    base_score: float,
    base_d_seg: float,
    base_d_pose: float,
    base_archive_bytes: int,
    cand_d_seg: float,
    cand_d_pose: float,
    cand_archive_bytes: int,
    denominator: int = RATE_DENOMINATOR_BYTES,
) -> dict:
    """Decompose a candidate-vs-base move into its three S legs and the exchange ratios.

    Works for ANY candidate, not just this family -- the decomposition is just the score
    function. What makes it a law is the ratios it reports and the anchor they are read
    against.

    ``ratio_*`` fields are ``None`` when the candidate did not buy rate (credit <= 0);
    dividing by a non-credit would manufacture a meaningless number, and reporting the
    absence is the honest form.
    """
    getcontext().prec = 40
    den = Decimal(denominator)
    rate = lambda b: Decimal(25) * Decimal(b) / den  # noqa: E731
    pose = lambda dp: (Decimal(10) * _dec(dp)).sqrt()  # noqa: E731

    d_seg_leg = Decimal(100) * (_dec(cand_d_seg) - _dec(base_d_seg))
    d_pose_leg = pose(cand_d_pose) - pose(base_d_pose)
    d_rate_leg = rate(cand_archive_bytes) - rate(base_archive_bytes)
    net = d_seg_leg + d_pose_leg + d_rate_leg

    credit = -d_rate_leg
    has_credit = credit > 0
    return {
        "seg_leg_S": float(d_seg_leg),
        "pose_leg_S": float(d_pose_leg),
        "rate_leg_S": float(d_rate_leg),
        "net_S": float(net),
        "rate_credit_S": float(credit) if has_credit else None,
        "delta_bytes": cand_archive_bytes - base_archive_bytes,
        "delta_d_seg": float(_dec(cand_d_seg) - _dec(base_d_seg)),
        "delta_flips": float(Decimal(SEG_PIXELS) * (_dec(cand_d_seg) - _dec(base_d_seg))),
        "ratio_pose_damage": float(d_pose_leg / credit) if has_credit else None,
        "ratio_seg_damage": float(d_seg_leg / credit) if has_credit else None,
        "ratio_retained": float(-net / credit) if has_credit else None,
        "base_recomputed_S": float(
            Decimal(100) * _dec(base_d_seg) + pose(base_d_pose) + rate(base_archive_bytes)
        ),
        "cand_recomputed_S": float(
            Decimal(100) * _dec(cand_d_seg) + pose(cand_d_pose) + rate(cand_archive_bytes)
        ),
        "score_claim": False,
    }


def project_at_mass(
    mass: float,
    *,
    base_d_pose: float = BASE_D_POSE,
    unit_rate_credit_S: float = RATE_CREDIT_S,
    unit_seg_damage_S: float = SEG_DAMAGE_S,
    unit_delta_d_pose: float | None = None,
    residual_fraction_vs_unit: float = 1.0,
) -> dict:
    """Project the family's net S at ``mass`` x the sa3 edit, honoring the concavity.

    Rate credit and seg damage scale LINEARLY. ``d_pose`` scales linearly (sa1: 0.91x of
    linear, so this is mildly conservative); the pose S-leg is then re-derived by
    ``sqrt(10*d_pose)``, which is where the concavity enters and why retention RISES with
    mass.

    The single most common way to get this wrong is to scale the pose S-leg directly --
    that assumes linearity the score function does not have, and it under-states the
    family everywhere above mass 1.

    ``residual_fraction_vs_unit`` IS THE PRECONDITION, made explicit. It multiplies the
    per-unit residual ``d_pose`` delta, so 1.0 (the default) means "this successor's
    in-compile compensation cancels exactly as well as sa3's did". It is NOT a law:
    cancellation is re-solved per object and does not follow from mass. keep01 measured
    99.9801% cancellation at 3.0x mass where sa3's row implies less, which is why its
    realized retention was 42.2% against ~11.9% predicted at the default -- a 3.3x
    under-prediction (see KEEP01_* above).

    Pass a MEASURED value whenever the successor's cancellation is known. Leaving the
    default on an object whose compensation has NOT been re-solved is the cross-regime
    constant transfer that cost qs4 +2.396e-04.
    """
    if mass <= 0:
        raise ValueError("mass must be positive")
    if residual_fraction_vs_unit < 0:
        raise ValueError("residual_fraction_vs_unit must be non-negative")
    getcontext().prec = 40
    m = _dec(mass)
    dp_base = _dec(base_d_pose)
    dp_unit = (
        _dec(unit_delta_d_pose)
        if unit_delta_d_pose is not None
        else _dec(ANCHOR_D_POSE) - dp_base
    ) * _dec(residual_fraction_vs_unit)
    pose_leg = (Decimal(10) * (dp_base + m * dp_unit)).sqrt() - (Decimal(10) * dp_base).sqrt()
    seg_leg = m * _dec(unit_seg_damage_S)
    credit = m * _dec(unit_rate_credit_S)
    net = pose_leg + seg_leg - credit
    return {
        "mass": mass,
        "rate_credit_S": float(credit),
        "pose_damage_S": float(pose_leg),
        "seg_damage_S": float(seg_leg),
        "net_S": float(net),
        "retention_fraction": float(-net / credit),
        "projected_d_pose": float(dp_base + m * dp_unit),
        "pose_leg_is_concave_in_mass": True,
        "residual_fraction_vs_unit": residual_fraction_vs_unit,
        "assumes_sa3_compensation_quality": residual_fraction_vs_unit == 1.0,
        "score_claim": False,
    }


def family_cannot_close_alone(
    gap_S: float, *, residual_fraction_vs_unit: float = 1.0
) -> dict:
    """Mass (and rate-credit bytes) this family needs to close ``gap_S`` by itself.

    Returns the demand alongside the section that would have to supply it, so a caller
    sees the infeasibility rather than a bare multiplier. At the sz1 gap and sa3's
    compensation quality the demand is 1.20x the ENTIRE semantic section.

    THE CEILING IS CONDITIONAL ON COMPENSATION QUALITY, not on the family. It inherits
    ``project_at_mass``'s precondition, so ``residual_fraction_vs_unit`` moves it: keep01
    measured 3.3x better retention than the default predicts, which shrinks the demand
    correspondingly. Read ``feasible`` together with ``residual_fraction_vs_unit`` -- an
    infeasible verdict at the default is a statement about sa3-quality compensation, NOT a
    proof that no member of the family can route.
    """
    getcontext().prec = 40
    lo, hi = Decimal(1), Decimal(10_000)
    target = _dec(gap_S)
    for _ in range(200):
        mid = (lo + hi) / 2
        projected = project_at_mass(
            float(mid), residual_fraction_vs_unit=residual_fraction_vs_unit
        )
        if -_dec(projected["net_S"]) < target:
            lo = mid
        else:
            hi = mid
    mass = (lo + hi) / 2
    bytes_needed = float(mass) * abs(BASE_ARCHIVE_BYTES - ANCHOR_ARCHIVE_BYTES)
    return {
        "gap_S": gap_S,
        "mass_multiple": float(mass),
        "rate_credit_bytes_required": bytes_needed,
        "semantic_section_bytes": SEMANTIC_SECTION_BYTES,
        "fraction_of_semantic_section": bytes_needed / SEMANTIC_SECTION_BYTES,
        "feasible": bytes_needed <= SEMANTIC_SECTION_BYTES,
        "residual_fraction_vs_unit": residual_fraction_vs_unit,
        "ceiling_is_conditional_on_compensation_quality": True,
        "score_claim": False,
    }


def build_compensated_semantic_edit_exchange_v1() -> CanonicalEquation:
    provenance = build_provenance_for_research_sidecar(
        SOURCE_MEMO,
        reactivation_criteria=(
            "re-calibrate when any member of this family lands a T4 row at a materially "
            "different edit mass, or when the pointer's d_pose moves enough that the pose "
            "marginal 5/sqrt(10*d_pose) shifts by more than ~10%; the SHAPE never needs "
            "recalibration, only the unit-mass legs do"
        ),
        measurement_axis=AXIS,
        hardware_substrate="linux_x86_64_t4",
        captured_at_utc="2026-08-18T06:00:00Z",
    )
    anchors = (
        EmpiricalAnchor(
            anchor_id="sa3_compensated_edit_leg_split_20260818",
            measurement_utc="2026-08-18T06:00:00Z",
            inputs={
                "base": {
                    "sha256": BASE_ARCHIVE_SHA256,
                    "archive_bytes": BASE_ARCHIVE_BYTES,
                    "S": BASE_SCORE,
                    "d_seg": BASE_D_SEG,
                    "d_pose": BASE_D_POSE,
                },
                "candidate": {
                    "sha256": ANCHOR_ARCHIVE_SHA256,
                    "archive_bytes": ANCHOR_ARCHIVE_BYTES,
                    "S": ANCHOR_SCORE,
                    "d_seg": ANCHOR_D_SEG,
                    "d_pose": ANCHOR_D_POSE,
                },
                "mechanism": (
                    "lossy semantic-FiLM quantization (rate) + in-compile Schur pose "
                    "compensation solved against the EDITED object (qs5's proven form, "
                    "re-based onto the live sz1 pointer)"
                ),
                "axis": "[contest-CUDA T4 n600]",
                "n_samples": 600,
                "gpu_t4_match": True,
            },
            predicted_output={
                "sealed_falsifiers": {
                    "F1_net_below_band": "net < -3.5e-6",
                    "F2_seg_within_2x_of_predicted": "predicted d_seg delta +1.72e-6",
                    "F3_pose_damage_under_bar": "pose damage < +5.226e-4 S",
                    "F4_clean_row": "passed=True, validation_errors=[]",
                }
            },
            empirical_output={
                "legs_S": {
                    "rate": -RATE_CREDIT_S,
                    "pose": POSE_DAMAGE_S,
                    "seg": SEG_DAMAGE_S,
                    "net": NET_S,
                },
                "ratios_of_credit": {
                    "pose_damage": POSE_DAMAGE_FRACTION_OF_CREDIT,
                    "seg_damage": SEG_DAMAGE_FRACTION_OF_CREDIT,
                    "retained": RETENTION_FRACTION_AT_UNIT_MASS,
                },
                "falsifiers": {
                    "F1": "PASS — net -5.506e-05 is 15.7x the -3.5e-6 band",
                    "F2": "PASS — measured d_seg delta +2.04e-6 = 1.19x predicted, under 2x",
                    "F3": "PASS — pose damage +2.670e-4 consumed 51% of the rate credit",
                    "F4": "PASS — passed=True, rc=0, validation_errors=[]",
                },
                "verdict": "ADMITTED — eighth pointer move, 0.71% of the sub-0.15 gap",
                "family_ceiling": {
                    "mass_to_close_gap_alone": GAP_CLOSING_MASS_MULTIPLE,
                    "rate_credit_bytes_required": GAP_CLOSING_RATE_CREDIT_BYTES,
                    "semantic_section_bytes": SEMANTIC_SECTION_BYTES,
                    "verdict": (
                        "INFEASIBLE — the demand is 1.20x the entire semantic section, so "
                        "the family contributes but cannot route"
                    ),
                },
            },
            residual=0.0,
            source_artifact="/Volumes/APDataStore/pact/ddm_sa3/t4_row_cuda/MODAL_REMOTE_RESULT.json",
            measurement_method=(
                "archive.zip -> inflate.sh -> upstream/evaluate.py --device cuda on a "
                "Tesla T4, n=600, gpu_t4_match=True. Legs recomputed at 40-digit decimal "
                "precision from the harness's reported components; their sum reproduces "
                "the harness's own score_recomputed_from_components to 17 digits."
            ),
            provenance=provenance,
            empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
        ),
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name=(
            "Compensated lossy-semantic-edit exchange rate: what a byte of rate credit "
            "costs in pose and seg, and why the family gets more efficient with mass"
        ),
        one_line_summary=(
            "per S of rate credit: 0.508 back as pose, 0.388 as seg, 0.105 net; retention "
            "RISES with mass (concave pose leg); the gap alone needs 1.20x the whole "
            "semantic section — contributes, never routes"
        ),
        latex_form=(
            r"\Delta S(m)=\Big[\sqrt{10\,(d_p^{0}+m\,\delta_p)}-\sqrt{10\,d_p^{0}}\Big]"
            r"+m\,\Delta S_{seg}-m\,\Delta S_{rate}"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.compensated_semantic_edit_exchange_20260818"
            ":project_at_mass"
        ),
        domain_of_validity={
            "axis": AXIS,
            "research_only": False,
            "applies_to": (
                "members of the compensated lossy-semantic-edit family: coarser semantic "
                "quantization for rate, repaired by an in-compile Schur pose compensation "
                "solved against the EDITED object"
            ),
            "does_not_apply_to": (
                "UNCOMPENSATED lossy semantic edits — sa1 refused that family 3/3 with "
                "pose damage 68-512x the rate credit. The compensation is what makes the "
                "ratio 0.508 instead of two orders of magnitude worse. Also does not apply "
                "to a compensation carried from a DIFFERENT object: qs4 did exactly that "
                "and paid +2.396e-4 (cross-regime constant transfer)."
            ),
            "ratios_are_instance_scoped": (
                "one measured member at one operating point; the SHAPE (rate/seg linear, "
                "pose concave, retention rising with mass) is DERIVED-EXACT and carries "
                "further than the ratios do"
            ),
            "extrapolation_rule": (
                "scale d_pose, THEN take sqrt(10*d_pose). Scaling the pose S-leg directly "
                "assumes a linearity the score function does not have and under-states "
                "the family at every mass above 1."
            ),
        },
        units_in={
            "mass": "multiple of the sa3 edit (dimensionless)",
            "base_d_pose": "PoseNet MSE over the first 6 pose dims",
            "unit_rate_credit_S": "S",
            "unit_seg_damage_S": "S",
        },
        units_out={
            "net_S": "S",
            "retention_fraction": "dimensionless",
            "rate_credit_bytes_required": "bytes",
        },
        empirical_anchors=anchors,
        predicted_vs_empirical_residual={"registration": 0.0},
        last_calibration_utc="2026-08-18T06:00:00Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "any successor rung of the compensated-edit family: call project_at_mass() "
            "for its expected net before sealing, instead of assuming linear scaling",
            "fire-order falsifier bands for semantic-quantization candidates (the pose "
            "bar is a fraction of the credit, not an absolute)",
            "routing decisions between axes — family_cannot_close_alone() is why this "
            "family is composed rather than pursued to the gap",
            "ddm_asym1's three-axis asymmetry: this is the worked case where concavity "
            "helps rather than hurts, because the damage is paid upward",
        ),
        canonical_producers=(
            SOURCE_MEMO,
            "/Volumes/APDataStore/pact/ddm_sa3/t4_row_cuda/MODAL_REMOTE_RESULT.json",
            ".omx/research/ddm_asym1_three_axis_asymmetry_and_dynamics_20260818.md",
            ".omx/research/ddm_sa1_uncompensated_semantic_edit_family_refused_20260818.md "
            "(the uncompensated control: 68-512x, 3/3 refused)",
        ),
        provenance=provenance,
    )


def populate_compensated_semantic_edit_exchange_equation(
    *, path=None, lock_path=None, agent: str | None = None, subagent_id: str | None = None
) -> CanonicalEquation:
    """Idempotent APPEND-ONLY registration."""
    from tac.canonical_equations.registry import register_canonical_equation

    eq = build_compensated_semantic_edit_exchange_v1()
    register_canonical_equation(
        eq, path=path, lock_path=lock_path, agent=agent, subagent_id=subagent_id
    )
    return eq


__all__ = [
    "ANCHOR_ARCHIVE_BYTES",
    "ANCHOR_ARCHIVE_SHA256",
    "ANCHOR_D_POSE",
    "ANCHOR_D_SEG",
    "ANCHOR_SCORE",
    "AXIS",
    "BASE_ARCHIVE_BYTES",
    "BASE_ARCHIVE_SHA256",
    "BASE_D_POSE",
    "BASE_D_SEG",
    "BASE_SCORE",
    "EQUATION_ID",
    "GAP_CLOSING_MASS_MULTIPLE",
    "GAP_CLOSING_RATE_CREDIT_BYTES",
    "NET_S",
    "POSE_DAMAGE_FRACTION_OF_CREDIT",
    "POSE_DAMAGE_S",
    "RATE_CREDIT_S",
    "RATE_DENOMINATOR_BYTES",
    "RETENTION_FRACTION_AT_UNIT_MASS",
    "SA1_POSE_MASS_LINEARITY",
    "KEEP01_MASS_MULTIPLE",
    "KEEP01_CANCELLATION_ACHIEVED",
    "KEEP01_CANCELLATION_REQUIRED",
    "KEEP01_MEAN_RESIDUAL_D_POSE",
    "KEEP01_MAX_RESIDUAL_ALLOWED",
    "KEEP01_PROJECTED_NET_S",
    "KEEP01_MEASURED_RETENTION",
    "KEEP01_NAIVE_UNDERPREDICTION_FACTOR",
    "SEG_DAMAGE_FRACTION_OF_CREDIT",
    "SEG_DAMAGE_S",
    "SEMANTIC_SECTION_BYTES",
    "build_compensated_semantic_edit_exchange_v1",
    "exchange_legs",
    "family_cannot_close_alone",
    "populate_compensated_semantic_edit_exchange_equation",
    "project_at_mass",
]
