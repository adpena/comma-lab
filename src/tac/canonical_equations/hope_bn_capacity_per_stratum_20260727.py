# SPDX-License-Identifier: MIT
"""Canonical law: HOPE per-neuron capacity on the frozen SegNet, exact n600 measure.

Registers ``hope_bn_capacity_per_stratum_codebook_v1`` — the analytic
generator law for the FISHER_MARGIN_SITE_LOCAL_PER_STRATUM_CODEBOOK
coordinate family (one of rg4's three demanded successor families):

    ||f_i||_H = ||w_out,i||_2 * sqrt(K(i,i)),     K(i,i) = E_P[Psi(y_i)^2]
    cap_b^{ab}(i) = ||Dw_head^{ab}[i]||_F * sqrt(K_b(i))

with P the EXACT n600 input measure (not HOPE's data-free Gaussian
surrogate; argmax agreement of the measurement pass with the cached GT
argmax was 0.999999991522895 — 1 px in 117,964,800), K_b the
stratum-restricted kernel over occupied pf2
bucket b, and Dw_head the class-pair difference of the exact rank-4
segmentation head (composes with ``segnet_head_rank4_linear_flipdist_v1``).

The paper's closed-form ReLU kernels (arXiv 2607.21366 Eqs. 3/79) apply
only to the decoder BN+ReLU units and are carried as a surrogate
comparison column; the encoder is SiLU + sigmoid SE gates (measured), so
no closed form is used anywhere as authority. Rate denominators are
deliberately absent: per the crosswalk caveat they must be measured coder
bytes, never parameter counts (score_units_per_byte_status=OWED).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    CanonicalEquation,
)
from tac.canonical_equations.evaluators import register_evaluator
from tac.provenance.builders import build_provenance_for_research_sidecar

EQUATION_ID = "hope_bn_capacity_per_stratum_codebook_v1"
REPO = Path(__file__).resolve().parents[3]
RECEIPT = REPO / (
    ".omx/research/ddm_hb1_hope_bn_capacity_20260727T0001Z/hope_rg3_agreement_receipt.json"
)
HEAD_IN_CHANNELS = 16


def _validated_vector(values: Sequence[int | float], *, label: str) -> np.ndarray:
    if len(values) != HEAD_IN_CHANNELS:
        raise ValueError(f"{label} must contain exactly {HEAD_IN_CHANNELS} channel entries")
    arr = np.asarray([float(v) for v in values], dtype=np.float64)
    if not np.isfinite(arr).all() or (arr < 0).any():
        raise ValueError(f"{label} must contain nonnegative finite numbers")
    return arr


def hope_stratum_capacity(
    delta_w_head_norm: Sequence[int | float],
    k_diag_bucket: Sequence[int | float],
) -> dict[str, Any]:
    """Per-channel HOPE capacity of one class-pair stratum (pure closed form).

    ``delta_w_head_norm[i]`` = Frobenius norm of the class-pair difference of
    the segmentation-head weight slab reading pre-head channel ``i``;
    ``k_diag_bucket[i]`` = exact stratum-restricted kernel diagonal
    E[psi_i^2 | bucket]. Both come from measured tables, never from the
    Gaussian surrogate.
    """

    dw = _validated_vector(delta_w_head_norm, label="delta_w_head_norm")
    k = _validated_vector(k_diag_bucket, label="k_diag_bucket")
    cap = dw * np.sqrt(k)
    total = float(cap.sum())
    share = cap / total if total > 0.0 else np.zeros_like(cap)
    return {
        "capacity_per_channel": [float(v) for v in cap],
        "capacity_share": [float(v) for v in share],
        "total_capacity": total,
        "dead_channels_k_lt_1e-12": int((k < 1e-12).sum()),
    }


def _evaluate(inputs: Mapping[str, Any]) -> dict[str, Any]:
    required = {"delta_w_head_norm", "k_diag_bucket"}
    if set(inputs) != required:
        raise ValueError("HOPE stratum-capacity inputs differ from the canonical callable contract")
    return hope_stratum_capacity(inputs["delta_w_head_norm"], inputs["k_diag_bucket"])


register_evaluator(EQUATION_ID, _evaluate)


def build_hope_bn_capacity_per_stratum_codebook_v1(
    *,
    source_receipt: Path = RECEIPT,
) -> CanonicalEquation:
    """Build the HOPE exact-measure per-stratum capacity law."""

    provenance = build_provenance_for_research_sidecar(
        source_receipt,
        reactivation_criteria=(
            "Rebuild after any SegNet checkpoint, pf2 bucket index, gt_n600 cache, "
            "head factorization, or margin-field SHA changes, or if a measured "
            "coder-byte denominator is admitted for any capacity row."
        ),
        measurement_axis="[macOS-CPU frozen-scorer advisory]",
        hardware_substrate="darwin_arm64_cpu_torch_fp32_batch4",
        captured_at_utc="2026-07-27T00:01:00Z",
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="HOPE BN-capacity per stratum codebook weighting (exact n600 measure)",
        one_line_summary=(
            "Per-channel HOPE capacity ||Dw_head[i]||*sqrt(K_b(i)) with exact "
            "stratum-restricted empirical kernels generates the Fisher-margin "
            "site-local per-stratum codebook weighting."
        ),
        latex_form=(
            r"\|f_i\|_H=\|w_{\mathrm{out},i}\|_2\sqrt{K(i,i)},\quad "
            r"K_b(i)=\mathbb{E}_{P_{600}}[\psi_i^2\mid b],\quad "
            r"\mathrm{cap}_b^{ab}(i)=\|\Delta w^{ab}_{\mathrm{head}}[i]\|_F\sqrt{K_b(i)},\quad "
            r"W(x)=\textstyle\sum_i\hat c_i\,\psi_i(x)^2"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.hope_bn_capacity_per_stratum_20260727:hope_stratum_capacity"
        ),
        domain_of_validity={
            "scorer": "frozen SegNet smp.Unet tu-efficientnet_b2 classes=5 (SHA-pinned)",
            "measure": "exact n600 last-frame inputs; argmax agreement 0.999999991522895 vs cached GT (1 px / 117,964,800)",
            "strata": "37 occupied pf2 buckets (class pair x cell/boundary x temporal)",
            "head_composition": "exact rank-4 head (segnet_head_rank4_linear_flipdist_v1); rank check = 4",
            "relu_family_check": (
                "encoder SiLU + sigmoid SE gates => HOPE closed-form kernels apply to the "
                "10 decoder BN+ReLU units only, and only as a surrogate comparison column"
            ),
            "consumer_graph": (
                "capacities emitted only where w_out is structurally unambiguous; "
                "encoder block outputs carry UNRESOLVED_CONSUMER_GRAPH_V1 and sqrt(K) only"
            ),
            "rate_policy": "no rate columns; denominators must be measured coder bytes (OWED)",
            "research_only": True,
            "score_claim": False,
            "evidence_axis": "[macOS-CPU frozen-scorer advisory]",
            "verdict_scope": "INSTANCE_GENERATOR_FOR_FISHER_MARGIN_SITE_LOCAL_PER_STRATUM_CODEBOOK",
            "excluded": [
                "contest score or frontier movement",
                "HOPE paper effect sizes (never quotable as ours)",
                "Gaussian-surrogate kernels as authority",
                "parameter-count rate denominators",
                "witness-side adoption without a basis-specific kernel derivation",
            ],
        },
        units_in={
            "delta_w_head_norm": "Frobenius norm of head class-pair weight difference per pre-head channel",
            "k_diag_bucket": "exact second moment of pre-head activation restricted to the bucket",
        },
        units_out={
            "capacity_per_channel": "HS-norm capacity units (scorer-internal, dimensionless)",
            "capacity_share": "fraction of stratum capacity per channel",
            "total_capacity": "HS-norm capacity units",
        },
        empirical_anchors=(),
        predicted_vs_empirical_residual={},
        last_calibration_utc="2026-07-27T00:01:00Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.optimization.hope_bn_capacity",
            "tools.run_ddm_hb1_hope_bn_capacity",
        ),
        canonical_producers=(
            "tools.run_ddm_hb1_hope_bn_capacity",
        ),
        provenance=provenance,
    )


def populate_hope_bn_capacity_per_stratum_codebook_v1(
    *,
    source_receipt: Path = RECEIPT,
    path: Path | None = None,
    lock_path: Path | None = None,
    agent: str | None = None,
    subagent_id: str | None = None,
) -> CanonicalEquation:
    """Append the HOPE capacity law through the locked registry helper."""

    from tac.canonical_equations.registry import register_canonical_equation

    equation = build_hope_bn_capacity_per_stratum_codebook_v1(source_receipt=source_receipt)
    register_canonical_equation(
        equation,
        path=path,
        lock_path=lock_path,
        agent=agent,
        subagent_id=subagent_id,
        notes=(
            "HOPE exact-measure capacity generator; reproduces 17/17 RG3 Fisher rows; "
            "advisory only; score_claim=false; MAIN review required"
        ),
    )
    return equation


__all__ = [
    "EQUATION_ID",
    "build_hope_bn_capacity_per_stratum_codebook_v1",
    "hope_stratum_capacity",
    "populate_hope_bn_capacity_per_stratum_codebook_v1",
]
