# SPDX-License-Identifier: MIT
"""Regression tests for the recon_pixel_weight boundary-refinement registrar."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from tac.provenance import build_provenance_for_macos_cpu_advisory


def _load_registrar():
    repo_root = Path(__file__).resolve().parents[3]
    tool_path = repo_root / "tools" / "register_recon_pixel_weight_boundary_refinement.py"
    spec = importlib.util.spec_from_file_location("register_recon_pixel_weight_boundary_refinement", tool_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(*, d_seg_uniform: float, d_seg_saliency: float) -> dict:
    return {
        "n_levels": 6,
        "baseline_d_seg": 0.0125,
        "d_seg_uniform": d_seg_uniform,
        "d_seg_full_grid_saliency": d_seg_saliency,
        "d_seg_s_uniward_texture": 0.0037,
        "saliency_nonzero_fraction": 0.77,
        "saliency_beats_uniform": d_seg_saliency < d_seg_uniform,
        "verdict": "LEVER_NOT_CONFIRMED_ON_REAL_RENDER",
    }


def test_dense_no_transfer_records_sign_boundary_miss_not_zero_residual() -> None:
    mod = _load_registrar()
    prov = build_provenance_for_macos_cpu_advisory(
        archive_sha256="a" * 64,
        source_path="experiments/results/recon_pixel_weight_real_render_ab_20260531/ab_output.json",
        captured_at_utc="2026-05-31T17:30:00+00:00",
    )
    anchor = mod._build_boundary_refinement_anchor(
        row=_row(d_seg_uniform=0.001, d_seg_saliency=0.001),
        now="2026-05-31T17:30:00+00:00",
        source_artifact="experiments/results/recon_pixel_weight_real_render_ab_20260531/ab_output.json",
        provenance=prov,
    )

    assert anchor.predicted_output["margin_saliency_vs_uniform_sign"] == "positive"
    assert anchor.predicted_output["residual_type"] == "sign_boundary_miss"
    assert anchor.empirical_output["margin_saliency_vs_uniform"] == 0.0
    assert anchor.empirical_output["margin_saliency_vs_uniform_sign"] == "non_positive"
    assert anchor.empirical_output["margin_sign_match"] is False
    assert anchor.residual == 1.0
    assert "correction_concentration" not in anchor.empirical_output


def test_dense_positive_transfer_keeps_zero_sign_residual() -> None:
    mod = _load_registrar()
    prov = build_provenance_for_macos_cpu_advisory(
        archive_sha256="b" * 64,
        source_path="experiments/results/recon_pixel_weight_real_render_ab_20260531/ab_output.json",
        captured_at_utc="2026-05-31T17:30:00+00:00",
    )
    anchor = mod._build_boundary_refinement_anchor(
        row=_row(d_seg_uniform=0.002, d_seg_saliency=0.001),
        now="2026-05-31T17:30:00+00:00",
        source_artifact="experiments/results/recon_pixel_weight_real_render_ab_20260531/ab_output.json",
        provenance=prov,
    )

    assert anchor.empirical_output["margin_saliency_vs_uniform"] == 0.001
    assert anchor.empirical_output["margin_saliency_vs_uniform_sign"] == "positive"
    assert anchor.empirical_output["margin_sign_match"] is True
    assert anchor.residual == 0.0
