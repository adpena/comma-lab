"""ddm_cr1 — the seg-only-trained-base POSE DEGRADATION law (canonical equation + 3 anchors).

#827 asked whether the burn seg base composes with the live pose carrier.  The named blocker was
stated as a TRANSFER cost ("the pose was fitted to different pixels").  The matched-pair
measurement (memo ``.omx/research/ddm_cr1_composition_row_827_20260801.md``) REFUTES the transfer
framing and replaces it with a base-property law:

  * transfer per se is nearly free -- the pfs1 D1 warp shipped VERBATIM onto the gr1 cell_drop50
    base and the EXACT evaluator measured d_pose 0.234817 against the solve's own 0.221547, i.e.
    +6.0%.  Base swaps do not break the warp-base carrier.
  * a SEG-ONLY-TRAINED base does.  Re-solving the warp on the burn endpoint's OWN shipped frame_1
    (the strongest form the carrier admits) gave mean d_pose 3.112073 vs a matched-pair control of
    0.489332 on the same 61 pairs, same frozen PoseNet, same shipped f16 targets: 6.36x worse, on
    61 of 61 pairs.

Mechanism (DERIVED, and predicted by the standing CLAUDE.md pose clarification): ``tr1_config.json``
for that burn window carries ``w_seg 100.0`` and NO pose term of any kind, so the renderer was driven
purely toward SegNet-argmax fidelity -- a task-lossy objective free to destroy the photometric
texture PoseNet reads.  The solver's ``s_t`` choice corroborates: the control concentrates at grid
index 7 and the burn base shifts DOWN to index 6, both INTERIOR to the 11-point grid, so this is a
render property and not grid saturation.

The law this registers: ``pose_degradation_ratio`` = mean d_pose on a candidate base divided by the
matched-pair mean on a pose-neutral control base.  ~1 means the base is pose-transparent; the one
measured seg-only-trained base scores 6.36.

Pointer honesty: pointer UNMOVED.  Advisory; score_claim=False.  The anchors are measured d_pose
means from the frozen CPU PoseNet, never score or pointer claims.
"""

from __future__ import annotations

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    VERIFIED_VIA_EMPIRICAL_ANCHOR,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

EQUATION_ID = "ddm_cr1_seg_only_base_pose_degradation_v1"

SOURCE_ARTIFACT = ".omx/research/ddm_cr1_composition_row_827_20260801.md"

# MEASURED, matched pairs 0..60, same tool/targets/protocol, only the seg base differs.
CONTROL_D_POSE_N61 = 0.489332  # p2c_aimed (pose-neutral base; no pose term either, but not burned)
SEG_ONLY_BASE_D_POSE_N61 = 3.112073  # b4s window_03 ep854 (w_seg 100.0, zero pose term)
# MEASURED, exact evaluator: the VERBATIM transfer control (p2c_aimed warp -> gr1 cell_drop50 base).
TRANSFER_CONTROL_SOLVE_D_POSE_N600 = 0.221547
TRANSFER_CONTROL_EXACT_D_POSE_N600 = 0.23481703

# MEASURED 2026-08-02 (ddm_cr2r) -- a SECOND, INDEPENDENT carrier class on the same ep854 base.
# Not the warp-base carrier: this is the v4c STATIC TWO-PLANE solve (``--mode solve``), a FRESH
# FULL re-solve rather than a warp re-fit, controlled against the celldrop50 base.  74 matched
# pairs, same tool, same code path, same fields; the ONLY variable is the base archive.
# The ratio is ~23x LARGER than the warp-base carrier's 6.36 -- the degradation is carrier-
# dependent in MAGNITUDE while the SIGN and the base-property mechanism replicate.
CR2R_SEG_ONLY_BASE_D_POSE_N74 = 11.5904  # ep854 (same seg-only burn base as the n61 anchor)
CR2R_CONTROL_D_POSE_N74 = 0.0778  # celldrop50 (pose-neutral control base)
# Difficulty-profile self-control, same shape as the n61 anchor: these 74 pairs are the
# hardest-first-by-KneeA set, and the control's mean on them sits above its own solve-stage
# n600 mean -- so, again, only the matched-pair RATIO transfers, never the LEVEL.
CR2R_CONTROL_D_POSE_N600_SOLVE_STAGE = 0.0273


def pose_degradation_ratio(candidate_d_pose: float, control_d_pose: float) -> float:
    """Ratio of matched-pair mean d_pose on a candidate base to a control base.

    ~1.0 => the base is pose-transparent (the carrier survives the swap).
    >>1.0 => the base's renders are photometrically hostile to a render-reading pose carrier.
    """
    control = float(control_d_pose)
    if control <= 0.0:
        raise ValueError("control_d_pose must be positive")
    return float(candidate_d_pose) / control


def pose_contribution(d_pose_mean: float) -> float:
    """The score's pose term sqrt(10*d_pose) -- the units the ratio must be read in."""
    return float(10.0 * float(d_pose_mean)) ** 0.5


def _anchor(anchor_id, candidate, control, recorded, method, source, provenance,
            measurement_utc="2026-08-01T00:00:00Z"):
    ratio = pose_degradation_ratio(candidate, control)
    return EmpiricalAnchor(
        anchor_id=anchor_id,
        measurement_utc=measurement_utc,
        inputs={"candidate_d_pose": candidate, "control_d_pose": control},
        predicted_output={"pose_degradation_ratio": recorded},
        empirical_output={"pose_degradation_ratio": ratio},
        residual=abs(ratio - recorded),
        source_artifact=source,
        measurement_method=method,
        provenance=provenance,
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
        noise_floor=None,
        noise_floor_provenance=None,
    )


def build_seg_only_base_pose_degradation_v1() -> CanonicalEquation:
    """Build the degradation law + its three measured anchors."""
    provenance = build_provenance_for_research_sidecar(
        sidecar_path=SOURCE_ARTIFACT,
        reactivation_criteria=(
            "append an anchor for every new base measured under the matched-pair protocol; "
            "re-open the FORMULATION negative the moment a base trained WITH a nonzero pose "
            "term is measured, or a pose carrier that does not read the renders is tested; "
            "the 6.36 ratio is n61 single-seed -- across-seed variance is UNKNOWN"
        ),
        measurement_axis="[macOS-CPU advisory]",
        hardware_substrate="apple_macos_cpu_torch",
    )
    anchors = (
        _anchor(
            "cr1_segonly_ep854_vs_p2c_matched_n61_20260801",
            SEG_ONLY_BASE_D_POSE_N61, CONTROL_D_POSE_N61, 6.360,
            "matched-pair warp re-solve, pairs 0..60, frozen CPU PoseNet, shipped f16 targets; "
            "candidate = b4s window_03 ep854 TR1 archive sha 37ba7a96; worse on 61/61 pairs",
            "/Volumes/VertigoDataTier/pact/ddm_cr1_20260801/ep854_compose/"
            "d1_warp_solve.partial.jsonl",
            provenance,
        ),
        _anchor(
            "cr1_transfer_control_verbatim_warp_gr1_20260801",
            TRANSFER_CONTROL_EXACT_D_POSE_N600, TRANSFER_CONTROL_SOLVE_D_POSE_N600, 1.0599,
            "VERBATIM transfer control: the pfs1 D1 warp shipped unchanged onto the gr1 "
            "cell_drop50 base; numerator is the EXACT upstream/evaluate.py n600 d_pose "
            "(archive sha a6398e44, rc=0), denominator the solve's own n600 mean -- "
            "transfer alone costs +6.0%, so transfer is NOT the mechanism",
            "/Volumes/VertigoDataTier/pact/ddm_ep2_20260731/gr1_eval/d1_eval_receipt.json",
            provenance,
        ),
        _anchor(
            "cr1_self_control_p2c_n61_vs_n600_profile_20260801",
            CONTROL_D_POSE_N61, TRANSFER_CONTROL_SOLVE_D_POSE_N600, 2.2087,
            "difficulty-profile self-control on the SAME base: pairs 0..60 are 2.21x the "
            "control's own n600 mean, which is why the n61 candidate mean must not be read "
            "as an n600 mean (the RATIO is the transferable statistic, not the level)",
            "/Volumes/VertigoDataTier/pact/ddm_pfs1_20260729/d1/d1_warp_solve.partial.jsonl",
            provenance,
        ),
        _anchor(
            "cr2r_segonly_ep854_v4c_static_two_plane_vs_celldrop50_n74_20260802",
            CR2R_SEG_ONLY_BASE_D_POSE_N74, CR2R_CONTROL_D_POSE_N74, 148.977,
            "SECOND CARRIER CLASS, independent of the warp-base anchor: v4c static two-plane "
            "FRESH FULL re-solve (ddm_v4c_resolve --mode solve), 74 matched pairs, same tool and "
            "code path, only the base archive differs (ep854 sha fd509258 vs celldrop50); ep854 "
            "worse on 73/74.  The cr2 pose TRANSPLANT onto ep854 measured 37.877, so a fresh "
            "re-solve improves 3.3x and is still ~880x above the 0.0131903119638695 break-even.  "
            "The SIGN and base-property mechanism replicate the warp-base anchor; the MAGNITUDE "
            "does not (149 vs 6.36), so the ratio is carrier-dependent and must be re-measured "
            "per carrier rather than transferred",
            "/Volumes/VertigoDataTier/pact/ddm_v4c_20260730/solve_ep854.partial.jsonl",
            provenance,
            measurement_utc="2026-08-02T00:00:00Z",
        ),
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="seg-only-trained base pose degradation ratio",
        one_line_summary=(
            "pose carriers survive a base SWAP (+6.0%) but not a SEG-ONLY base: 6.36x warp-base "
            "(61/61), 148.98x v4c two-plane (73/74) -- rho is carrier-dependent; cr1+cr2r memos"
        ),
        latex_form=r"\rho=\frac{\overline{d_{pose}}^{\,cand}}{\overline{d_{pose}}^{\,ctrl}},\quad"
                   r"\Delta S_{pose}=\sqrt{10\,\overline{d_{pose}}^{\,cand}}"
                   r"-\sqrt{10\,\overline{d_{pose}}^{\,ctrl}}",
        python_callable_module_path=(
            "tac.canonical_equations.ddm_cr1_seg_only_base_pose_degradation_20260801:"
            "pose_degradation_ratio"),
        domain_of_validity={
            "included": [
                "render-reading pose carriers on TR1 seg endpoints -- TWO measured carrier "
                "classes: warp-base PFS1WPB1 (6.36x) and v4c static two-plane (148.98x)",
                "go/no-go on composing a seg-only-trained base with an existing pose carrier",
                "matched-pair comparisons only (same pair indices, same targets, same protocol)",
                "the SIGN and the base-property mechanism, which replicate across both carriers",
            ],
            "excluded": [
                "score, promotion, or pointer movement",
                "transferring the ratio MAGNITUDE across carrier classes -- measured 6.36 vs "
                "148.98 on the SAME base, so rho is carrier-dependent and must be re-measured "
                "per carrier; only the sign/mechanism generalize",
                "pose carriers that do NOT read the renders (untested class)",
                "bases trained WITH a nonzero pose term (untested; the named reformulation)",
                "reading the n61 candidate mean as an n600 mean -- the LEVEL does not transfer, "
                "only the matched-pair RATIO (see the difficulty-profile anchor)",
                "the terminal-GN carrier (state/pose.tpgn), a different section and separately "
                "confounded by the #850 relinearization<=3 truncation",
            ],
            "authority": "[macOS-CPU advisory]",
        },
        units_in={"candidate_d_pose": "PoseNet MSE over first 6 dims",
                  "control_d_pose": "PoseNet MSE over first 6 dims"},
        units_out={"pose_degradation_ratio": "dimensionless"},
        empirical_anchors=anchors,
        predicted_vs_empirical_residual={
            "max_anchor_residual": max(a.residual for a in anchors)},
        last_calibration_utc="2026-08-02T00:00:00Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "#827 composition-row go/no-go (.omx/research/ddm_cr1_composition_row_827_20260801.md)",
            "burn charter pose-term decision (a burn window with w_pose>0 is the direct test)",
        ),
        canonical_producers=(
            "tools.pfs1_recompose_warp_base_and_eval:run_solve",
        ),
        provenance=provenance,
    )


def populate_seg_only_base_pose_degradation_v1(
    *, path=None, lock_path=None, agent: str | None = None, subagent_id: str | None = None
) -> CanonicalEquation:
    """Append the degradation law through the locked registry helper."""

    from tac.canonical_equations.registry import register_canonical_equation

    equation = build_seg_only_base_pose_degradation_v1()
    register_canonical_equation(
        equation,
        path=path,
        lock_path=lock_path,
        agent=agent,
        subagent_id=subagent_id,
        notes=(
            "cr1 #827 seg-only-base pose degradation; [macOS-CPU advisory]; score_claim=false; "
            "FORMULATION-scoped negative on the warp-base carrier, family OPEN"
        ),
    )
    return equation
