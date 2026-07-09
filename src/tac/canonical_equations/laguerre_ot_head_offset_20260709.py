# SPDX-License-Identifier: MIT
"""Canonical equation: #288 damped-Newton SEMI-DISCRETE OT per-class head-bias offset (solve-don't-train
inventory row 2). The PRINCIPLED alternative to the Menon ``-tau*log(pi)`` prior heuristic.

The witness decodes ``argmax_k phi_k`` (K=5 SDF fields). Adding a per-class additive offset ``b`` gives
``argmax_k (phi_k + b_k)`` — a POWER DIAGRAM (Laguerre) reweight of the argmax cells (folds BYTE-FREE into
``out_sdf.bias``). Semi-discrete OPTIMAL TRANSPORT (Kitagawa-Merigot-Thibert 2019) solves the UNIQUE
zero-sum ``b*`` whose Laguerre-reweighted SOFT cell masses equal prescribed target masses ``pi``:

    dual   Phi(b) = <pi, b> - tau * mean_p logsumexp((phi_p + b)/tau)     (concave)
    grad   g(b)   = pi - m(b),   m_c(b) = mean_p softmax((phi_p+b)/tau)_c (soft cell masses)
    Hess   H(b)   = -(1/tau)(diag(m) - mean_p s_p s_p^T)                  (NSD softmax covariance, rank K-1)
    step   b <- b + t * tau * pinv(diag(m) - mean_p s_p s_p^T) @ (pi - m)  (zero-sum gauge; Armijo t)

H is rank K-1 (all-ones gauge nullspace) => zero-sum pseudo-inverse; the Armijo backtrack on the concave
dual guarantees global convergence with terminal QUADRATIC rate. Unlike Menon (which needs only priors and
IGNORES the witness logit geometry), the OT solve uses THIS witness's actual boundary geometry (the phi
field), so it accounts for the boundary lengths the log-freq heuristic cannot see.

THE MEASURED FINDING (the honest verdict, VERIFIED_VIA_EMPIRICAL_ANCHOR — realized THROUGH R on the frozen
CPU SegNet, mod32cap ep650 EMA): matching the Laguerre cell MASSES to the GT class frequencies is NOT the
same objective as minimizing realized d_seg. The OT solve enlarges the rare-Lane cell to hit its 0.59%
GT mass, but that OVER-PREDICTS Lane relative to the witness's own boundary placement, and the SegNet
re-read PENALISES the over-prediction — so the OT offset made realized d_seg WORSE than BOTH no-offset and
the Menon prior at this operating point. This is a genuine measured negative for the mass-matching
objective (verdict_scope: FORMULATION — "cell-mass-matching as a d_seg surrogate", at this checkpoint/tau;
NOT the OT solver's correctness, which is exact: it converges to max_mass_err 0 in <=8 Newton iters).

DEGENERACIES (VERIFIED_VIA_SOURCE_INSPECTION — the byte-identity contract): the trainer wire-in is an
ADVISORY verdict readout gated on ``--head-offset-solver != off`` (default ``off``); it folds b* into a
COPY of ``out_sdf.bias`` and NEVER touches the EMA shadow / shipped / resumed weights, so the default path
is byte-identical and the ON path leaves the authority trajectory unchanged (the offset changes only the
advisory row). A GLOBAL constant added to every logit changes no argmax => the offsets are mean-centered
(zero-sum), the canonical Laguerre-weight gauge.

means != ends: this law + the solver BUILD + MEASURE the mechanism; they make NO score claim. The gate is
[macOS-CPU advisory] realized-through-R, NON-PROMOTABLE until byte-closed exact eval; pointer 0.19110
UNMOVED.

DSL leg: ``tac.witness_dsl.curriculum_dsl.HeadOffsetSolver`` (--head-offset-solver {off,menon,ot_newton}
+ --head-offset-solver-tau). Mechanism + canonical selectable entry:
``tac.boundary_math.laguerre_logit_offset.{damped_newton_ot_offsets, solve_head_offsets}``. $0 gate:
``experiments/probe_laguerre_logit_offset_sweep.py`` (3-arm no-offset / menon / ot_newton).
"""
from __future__ import annotations

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    VERIFIED_VIA_EMPIRICAL_ANCHOR,
    VERIFIED_VIA_SOURCE_INSPECTION,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import (
    build_provenance_for_predicted,
    build_provenance_for_research_sidecar,
)

EQUATION_ID = "laguerre_ot_head_offset_v1"

_UTC = "2026-07-09T00:00:00Z"
_ADVISORY = "[contest-CPU advisory]"
_PREDICTED = "[predicted]"
_DAG = ".omx/research/sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md"

# --- the #288 $0 gate MEASURED anchor (realized-through-R, frozen CPU SegNet, mod32cap ep650) --------
# n48 realized d_seg, 3 apples-to-apples arms (folded byte-free into out_sdf.bias). COMPLETE; the verdict
# ORDER + sign REPRODUCES at n24 (no_offset 0.0027349 / menon 0.0029672 / ot_newton 0.0047860) => robust
# across two complete subsets. The n96/n600 gates were killed by the background-task ~5-min limit (the
# probe is not resumable-chunked => full n600 is OWED). MASS-MATCHING is a MEASURED NEGATIVE for d_seg:
# OT > menon > no_offset (higher = worse). Numbers pinned VERBATIM from the completed n48 gate JSON.
GATE_N = 48
GATE_D_SEG_NO_OFFSET = 0.0027252
GATE_D_SEG_MENON = 0.0029322
GATE_D_SEG_OT_NEWTON = 0.0048664
GATE_N24_OT_NEWTON = 0.0047860   # n24 confirmation arm (same order/sign)
GATE_OT_CONVERGED = True
GATE_OT_ITERS = 7
GATE_OT_MAX_MASS_ERR = 0.0


def build_laguerre_ot_head_offset_v1() -> CanonicalEquation:
    """Build the #288 damped-Newton semi-discrete OT head-offset law with its honest-tier anchors.

    Two anchors: (1) the SOLVER CORRECTNESS + byte-identity DEGENERACIES, VERIFIED_VIA_SOURCE_INSPECTION
    (exact convergence to target masses; default-off byte-identical); (2) the $0 gate d_seg RESULT,
    VERIFIED_VIA_EMPIRICAL_ANCHOR (realized-through-R: mass-matching made d_seg WORSE at mod32cap ep650 —
    verdict_scope FORMULATION, at this checkpoint/tau)."""

    anchor_solver = EmpiricalAnchor(
        anchor_id="laguerre_ot_head_offset_solver_correctness_20260709",
        measurement_utc=_UTC,
        inputs={
            "mechanism": "tac.boundary_math.laguerre_logit_offset.damped_newton_ot_offsets",
            "dispatcher": "tac.boundary_math.laguerre_logit_offset.solve_head_offsets",
            "tests": "src/tac/boundary_math/tests/test_solve_head_offsets_ot.py",
        },
        predicted_output={
            "claim": ("damped-Newton on the concave OT dual converges to soft cell masses == target "
                      "masses (max_mass_err -> 0) in <=~8 Newton iters (terminal quadratic); offsets "
                      "zero-sum (mean-centered gauge); ot_newton NEVER silently falls back to menon"),
        },
        empirical_output={
            "mass_matching_property": "solved masses match target to tol (test_ot_solve_matches_target_masses)",
            "zero_sum_gauge": "sum(b*)~0 (test_ot_offsets_zero_sum)",
            "no_silent_fallback": "ot_newton with phi=None RAISES (test_ot_newton_requires_phi_no_fake)",
            "menon_alias": "solve_head_offsets('menon') == menon_logit_adjustment_offsets (byte-equal)",
            "verdict": ("the OT solver is EXACT; the negative below is about the mass-matching OBJECTIVE "
                        "as a d_seg surrogate, NOT the solver"),
        },
        residual=0.0,
        source_artifact="src/tac/boundary_math/laguerre_logit_offset.py",
        measurement_method="source inspection + NO-FAKE unit tests ($0 numpy)",
        empirical_verification_status=VERIFIED_VIA_SOURCE_INSPECTION,
        provenance=build_provenance_for_research_sidecar(
            sidecar_path="src/tac/boundary_math/laguerre_logit_offset.py",
            reactivation_criteria=("re-verify convergence + zero-sum + no-fallback if the Newton step / "
                                   "dual / dispatcher change"),
            measurement_axis=_ADVISORY,
            hardware_substrate="apple_m5_max_cpu",
        ),
    )
    anchor_gate = EmpiricalAnchor(
        anchor_id="laguerre_ot_head_offset_dollar0_gate_mod32cap_ep650_20260709",
        measurement_utc=_UTC,
        inputs={
            "ckpt": "levelset_n600_witness_mod32cap_20260706T115554Z/levelset_witness_ema_BEST.npz (ep650)",
            "gate": "experiments/probe_laguerre_logit_offset_sweep.py (3-arm)",
            "n_pairs": GATE_N,
            "authority": "realized-through-R, frozen CPU SegNet (fp32 EMA render)",
        },
        predicted_output={
            "hypothesis": ("the geometry-aware OT mass-matching offset beats the priors-only Menon "
                           "heuristic on realized d_seg (memo #288 row-2 HIGHEST-$0)"),
        },
        empirical_output={
            "d_seg_no_offset": GATE_D_SEG_NO_OFFSET,
            "d_seg_menon": GATE_D_SEG_MENON,
            "d_seg_ot_newton": GATE_D_SEG_OT_NEWTON,
            "ot_converged": GATE_OT_CONVERGED,
            "ot_max_mass_err": GATE_OT_MAX_MASS_ERR,
            "verdict": ("MEASURED-NEGATIVE: OT (mass-matching) > menon > no_offset (higher = worse). "
                        "Cell-mass-matching over-inflates the rare Lane cell => SegNet-re-read penalises "
                        "the over-prediction. Both offset arms HURT d_seg at this checkpoint; no-offset "
                        "wins. hypothesis REFUTED at mod32cap ep650."),
            "verdict_scope": ("FORMULATION — 'cell-mass-matching to GT frequencies as a d_seg surrogate' "
                              "at THIS checkpoint/tau. NOT the OT solver's correctness (exact). Other "
                              "checkpoints / tau / target-mass definitions (e.g. flip-weighted, not raw "
                              "frequency) UNTESTED = ASSUMED_AWAITING_VERIFICATION."),
            "advisory": "[contest-CPU advisory] NON-PROMOTABLE; pointer 0.19110 UNMOVED",
        },
        residual=0.0,
        source_artifact=_DAG,
        measurement_method="$0 offline gate: 3-arm realized-through-R d_seg (frozen CPU SegNet), n48 "
                            "(order+sign reproduced at n24; n96/n600 owed — background-task limit killed "
                            "them; probe not resumable-chunked)",
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
        provenance=build_provenance_for_research_sidecar(
            sidecar_path="experiments/probe_laguerre_logit_offset_sweep.py",
            reactivation_criteria=("re-run at other converged checkpoints, other tau, or a "
                                   "flip-weighted target-mass definition before any promotion"),
            measurement_axis=_ADVISORY,
            hardware_substrate="apple_m5_max_cpu",
        ),
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name=("#288 damped-Newton semi-discrete OT per-class head-bias offset: solve the zero-sum b* "
              "whose Laguerre-reweighted soft cell masses equal the GT class frequencies "
              "(Kitagawa-Merigot-Thibert 2019); byte-free out_sdf.bias fold"),
        one_line_summary=(
            "Semi-discrete OT solves the byte-free per-class argmax-cell offset to match GT masses; "
            "solver EXACT but mass-matching MEASURED-worse than no-offset for realized d_seg (ep650)."
        ),
        latex_form=(
            r"b^*=\arg\max_b\ \langle\pi,b\rangle-\tau\,\overline{\mathrm{lse}}\!\left(\tfrac{\phi+b}{\tau}\right),"
            r"\quad b\leftarrow b+t\,\tau\,(\mathrm{diag}(m)-\overline{ss^\top})^{+}(\pi-m),\ \ \bar b=0"
        ),
        python_callable_module_path=(
            "tac.boundary_math.laguerre_logit_offset:damped_newton_ot_offsets"
        ),
        domain_of_validity={
            "vehicle": ["softmax_of_sdf_levelset_witness", "coord_inr_seg_witness"],
            "lever": ("tac.witness_dsl.curriculum_dsl.HeadOffsetSolver "
                      "(--head-offset-solver {off,menon,ot_newton} / -tau)"),
            "measurement_axis": ["contest-CPU advisory", "predicted"],
            "note": ("the OT SOLVE is exact (mass-matching, zero-sum, quadratic Newton); the d_seg "
                     "VERDICT is a MEASURED NEGATIVE for the mass-matching objective at mod32cap ep650 "
                     "(verdict_scope FORMULATION). NON-PROMOTABLE until byte-closed exact eval."),
        },
        units_in={"phi": "witness_sdf_logits", "target_masses": "gt_class_frequencies",
                  "tau": "softmax_temperature"},
        units_out={"offsets": "per_class_additive_logit_bias (folds into out_sdf.bias, byte-free)"},
        empirical_anchors=(anchor_solver, anchor_gate),
        predicted_vs_empirical_residual={
            "solver_correctness_source_verified": 0.0,
            # the memo hypothesis (OT beats menon) is REFUTED: signed d_seg gap OT-vs-noOffset > 0 (worse).
            "gate_ot_minus_no_offset": GATE_D_SEG_OT_NEWTON - GATE_D_SEG_NO_OFFSET,
            "gate_menon_minus_no_offset": GATE_D_SEG_MENON - GATE_D_SEG_NO_OFFSET,
        },
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.witness_dsl.curriculum_dsl",       # DSL leg: HeadOffsetSolver factory
            "experiments/train_levelset_witness_realized_through_R_mlx.py",  # advisory verdict readout
            "experiments/probe_laguerre_logit_offset_sweep.py",             # the $0 gate
        ),
        canonical_producers=(
            "tac.boundary_math.laguerre_logit_offset",
        ),
        provenance=build_provenance_for_predicted(
            model_id="laguerre_ot_head_offset.v1",
            inputs_sha256="0" * 64,
            measurement_axis=_PREDICTED,
            hardware_substrate="apple_m5_max_cpu",
        ),
    )


def populate_laguerre_ot_head_offset_equation(
    *,
    path=None,
    lock_path=None,
    agent: str | None = None,
    subagent_id: str | None = None,
) -> CanonicalEquation:
    """Idempotent APPEND-ONLY registration of the #288 damped-Newton OT head-offset law (latest-row-wins).
    EQUATIONS leg of the #288 solver wire-in; DSL leg = ``curriculum_dsl.HeadOffsetSolver``; mechanism =
    ``tac.boundary_math.laguerre_logit_offset.damped_newton_ot_offsets``."""
    from tac.canonical_equations.registry import register_canonical_equation

    eq = build_laguerre_ot_head_offset_v1()
    register_canonical_equation(
        eq, path=path, lock_path=lock_path, agent=agent, subagent_id=subagent_id,
        notes="laguerre_ot_head_offset_20260709 (equations leg of the #288 OT head-offset wire-in; DSL "
              "leg = HeadOffsetSolver; $0 gate MEASURED mass-matching worse than no-offset at ep650)",
    )
    return eq


__all__ = [
    "EQUATION_ID",
    "GATE_D_SEG_MENON",
    "GATE_D_SEG_NO_OFFSET",
    "GATE_D_SEG_OT_NEWTON",
    "GATE_N",
    "build_laguerre_ot_head_offset_v1",
    "populate_laguerre_ot_head_offset_equation",
]
