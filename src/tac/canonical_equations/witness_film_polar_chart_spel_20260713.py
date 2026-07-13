# SPDX-License-Identifier: MIT
"""Canonical single-loop FiLM polar-chart MCSD/SPEL update law.

This equation refines the implementation surface of
``witness_modular_norm_assignment_v1``.  It does not replace that module's
still-open exact nested tangent-dual ticket: the law here is the explicitly
named, fireable MCSD/SPEL approximation shipped by Manifold-Muon round 2.
"""

from __future__ import annotations

import hashlib

import numpy as np

from tac.canonical_equations.equation import RECALIBRATE_ON_NEW_ANCHORS, CanonicalEquation
from tac.optimization.film_polar_chart_spel_mlx import tangent_project_numpy
from tac.provenance.builders import build_provenance_for_predicted

EQUATION_ID = "witness_film_polar_chart_spel_v1"
METHOD_ID = "film_polar_chart_mcsd_spel_v1"
_UTC = "2026-07-13T17:50:00Z"
_LAW = (
    "W=QH0; GQ=GW H0.T; TQ(X)=X-Q sym(Q.T X); "
    "M0=TQ(MW H0.T); M=TQ(beta M+(1-beta)TQ(GQ)); "
    "D=TQ(NS5((1-beta)TQ(GQ)+beta M))/max(1,||D||2); "
    "eta_eff=eta sqrt(max(1,rows/cols)); Q+=qf(Q-eta_eff D); M+=TQ+(M); W+=Q+H0"
)


def chart_reconstruction_relative_fro(weight: np.ndarray) -> float:
    """Return relative reconstruction error after a thin polar factorization."""

    from tac.optimization.film_polar_chart_spel_mlx import polar_chart_numpy

    w = np.asarray(weight, dtype=np.float32)
    q, h0 = polar_chart_numpy(w)
    denominator = max(float(np.linalg.norm(w, ord="fro")), np.finfo(np.float32).tiny)
    return float(np.linalg.norm(q @ h0 - w, ord="fro") / denominator)


def stiefel_tangent_residual_fro(q: np.ndarray, tangent: np.ndarray) -> float:
    """Return ``||Q^T A + A^T Q||_F`` after canonical projection."""

    qn = np.asarray(q, dtype=np.float32)
    projected = tangent_project_numpy(qn, tangent)
    return float(np.linalg.norm(qn.T @ projected + projected.T @ qn, ord="fro"))


def build_witness_film_polar_chart_spel_v1() -> CanonicalEquation:
    """Build the source-derived, empirically unpromoted fallback law."""

    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="FiLM polar-chart single-loop MCSD/SPEL Manifold-Muon fallback",
        one_line_summary=(
            "Freeze the positive polar factor H0, update only Q on Stiefel with a projected "
            "Muon matrix-sign direction, transport momentum, retract by deterministic QR, and fold W=QH0."
        ),
        latex_form=(
            r"W=QH_0,\ G_Q=G_WH_0^\top,\ \Pi_Q(X)=X-Q\operatorname{sym}(Q^\top X),\ "
            r"D=\Pi_Q(\operatorname{NS5}(\widetilde M))/\max(1,\|\Pi_Q(\operatorname{NS5}(\widetilde M))\|_2),\ "
            r"\eta_{eff}=\eta\sqrt{\max(1,m/n)},\ "
            r"Q^+=\operatorname{qf}(Q-\eta_{eff}D),\ M^+=\Pi_{Q^+}(M),\ W^+=Q^+H_0"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.witness_film_polar_chart_spel_20260713:"
            "chart_reconstruction_relative_fro"
        ),
        domain_of_validity={
            "architecture": "single film.weight tall full-column-rank V9 chart; mod_dim=19",
            "implementation_status": "FIREABLE_DEFAULT_OFF",
            "approximation_status": "MCSD/SPEL single-loop fallback; NOT exact nested tangent-dual LMO",
            "determinism": "seed-independent pure update given deterministic gradient stream",
            "resume_state": "Q, frozen H0, tangent momentum, Q-EMA, step, source weight SHA-256",
            "boundary_momentum": "outgoing AdamW first moment pulled through H0.T and tangent-projected",
            "learning_rate_convention": "matches MLX Muon sqrt(rows/cols) scale for tall film.weight",
            "measurement_status": (
                "NumPy-fp32 manifold and split-resume verified; MLX parity and local fine-tune require Metal"
            ),
            "verdict_scope": (
                "optimizer geometry and persistence only; no n600 d_seg, d_pose, byte, or score verdict"
            ),
        },
        units_in={
            "Q": "dimensionless conditioning isometry",
            "H0": "frozen FiLM weight units",
            "eta": "Stiefel spectral step units",
        },
        units_out={"W": "FiLM weight units"},
        empirical_anchors=(),
        predicted_vs_empirical_residual={},
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.optimization.film_polar_chart_spel_mlx",
            "tac.witness_dsl.curriculum_dsl.FilmPolarChartSPELManifoldMuon",
            "experiments.train_levelset_witness_realized_through_R_mlx",
            ".omx/research/muon_round2_wire_fireable_20260713.md",
            ".omx/research/muon_round2_wire_fireable_DAG_FEED_20260713.md",
        ),
        canonical_producers=(
            "tac.canonical_equations.witness_modular_norm_assignment_20260713",
            "https://thinkingmachines.ai/blog/modular-manifolds/",
        ),
        provenance=build_provenance_for_predicted(
            model_id=EQUATION_ID,
            inputs_sha256=hashlib.sha256(_LAW.encode("utf-8")).hexdigest(),
            measurement_axis="[source-derived plus numpy-fp32 local verification; non-promotable]",
            hardware_substrate="numpy-portable reference law",
            captured_at_utc=_UTC,
        ),
    )


__all__ = [
    "EQUATION_ID",
    "METHOD_ID",
    "build_witness_film_polar_chart_spel_v1",
    "chart_reconstruction_relative_fro",
    "stiefel_tangent_residual_fro",
]
