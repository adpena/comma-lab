# SPDX-License-Identifier: MIT
"""Smoke tests for tools/master_gradient_xray.py — 5 canonical plot types.

Per Cable D D4 (task #797) lane
`lane_cable_d_master_gradient_extension_batch_20260519`. Each plot type
gets a smoke test verifying it produces a valid PNG file with synthetic
input.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO_ROOT / "tools" / "master_gradient_xray.py"


@pytest.fixture(scope="module")
def xray_module():
    """Load tools/master_gradient_xray.py as a module."""
    spec = importlib.util.spec_from_file_location(
        "master_gradient_xray_test_module", TOOL_PATH
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["master_gradient_xray_test_module"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def synthetic_per_pair():
    rng = np.random.default_rng(7)
    return rng.normal(0.0, 1.0, size=(128, 12, 3)).astype(np.float64)


@pytest.fixture
def synthetic_aggregate(synthetic_per_pair):
    return synthetic_per_pair.mean(axis=1)


@pytest.fixture
def anchor_basic():
    return {
        "archive_sha256": "a" * 64,
        "measurement_axis": "contest_cuda",
        "measurement_hardware": "linux_x86_64_t4",
        "evidence_grade": "contest-CUDA",
    }


@pytest.fixture
def anchor_with_sections():
    return {
        "archive_sha256": "b" * 64,
        "measurement_axis": "contest_cpu",
        "measurement_hardware": "linux_x86_64_cpu",
        "evidence_grade": "contest-CPU",
        "archive_layout": {
            "sections": [
                {"name": "header", "offset": 0, "length": 32},
                {"name": "decoder", "offset": 32, "length": 48},
                {"name": "latent", "offset": 80, "length": 48},
            ]
        },
    }


@pytest.fixture
def anchor_mps_advisory():
    return {
        "archive_sha256": "c" * 64,
        "measurement_axis": "contest_cuda",
        "measurement_hardware": "darwin_arm64_m5_max_mps",
        "evidence_grade": "MPS-PROXY advisory",
    }


def test_plot_per_pair_distribution_produces_png(
    xray_module, synthetic_per_pair, anchor_basic, tmp_path
):
    out = tmp_path / "per_pair.png"
    xray_module.plot_per_pair_distribution(synthetic_per_pair, anchor_basic, out)
    assert out.exists()
    assert out.stat().st_size > 1000  # non-trivial PNG


def test_plot_per_byte_heatmap_produces_png(
    xray_module, synthetic_aggregate, anchor_basic, tmp_path
):
    out = tmp_path / "heatmap.png"
    xray_module.plot_per_byte_heatmap(
        synthetic_aggregate, anchor_basic, out, top_k=64
    )
    assert out.exists()
    assert out.stat().st_size > 1000


def test_plot_cumulative_by_rank_produces_png(
    xray_module, synthetic_aggregate, anchor_basic, tmp_path
):
    out = tmp_path / "cumulative.png"
    xray_module.plot_cumulative_by_rank(synthetic_aggregate, anchor_basic, out)
    assert out.exists()
    assert out.stat().st_size > 1000


def test_plot_cross_substrate_correlation_single_substrate_degenerate(
    xray_module, synthetic_aggregate, anchor_basic, tmp_path
):
    out = tmp_path / "correlation_single.png"
    xray_module.plot_cross_substrate_correlation(
        [("substrate_a", synthetic_aggregate, anchor_basic)], out
    )
    assert out.exists()  # Should emit placeholder PNG


def test_plot_cross_substrate_correlation_multi_substrate(
    xray_module, synthetic_aggregate, anchor_basic, tmp_path
):
    out = tmp_path / "correlation_multi.png"
    # Synthesize 3 substrates with varying correlation profiles
    rng = np.random.default_rng(13)
    a2 = rng.normal(0.0, 1.0, size=(128, 3))
    a3 = synthetic_aggregate + 0.1 * rng.normal(0.0, 1.0, size=(128, 3))  # similar to basic
    anchor_b = {**anchor_basic, "archive_sha256": "d" * 64}
    anchor_c = {**anchor_basic, "archive_sha256": "e" * 64}
    xray_module.plot_cross_substrate_correlation(
        [
            ("substrate_a", synthetic_aggregate, anchor_basic),
            ("substrate_b", a2, anchor_b),
            ("substrate_c", a3, anchor_c),
        ],
        out,
    )
    assert out.exists()
    assert out.stat().st_size > 1000


def test_plot_wyner_ziv_flow_with_sections(
    xray_module, synthetic_aggregate, anchor_with_sections, tmp_path
):
    out = tmp_path / "wz_flow.png"
    xray_module.plot_wyner_ziv_flow(synthetic_aggregate, anchor_with_sections, out)
    assert out.exists()
    assert out.stat().st_size > 1000


def test_plot_wyner_ziv_flow_without_sections_degrades_gracefully(
    xray_module, synthetic_aggregate, anchor_basic, tmp_path
):
    out = tmp_path / "wz_flow_no_sections.png"
    xray_module.plot_wyner_ziv_flow(synthetic_aggregate, anchor_basic, out)
    assert out.exists()
    assert out.stat().st_size > 500


def test_watermark_advisory_grade_marked(xray_module, anchor_mps_advisory):
    watermark = xray_module._watermark_for_anchor(anchor_mps_advisory)
    assert "advisory" in watermark.lower()


def test_watermark_authoritative_grade_unmarked(xray_module, anchor_basic):
    watermark = xray_module._watermark_for_anchor(anchor_basic)
    assert "advisory" not in watermark.lower()


def test_short_sha_handles_missing_or_short_input(xray_module):
    assert xray_module._short_sha(None) == "unknown"
    assert xray_module._short_sha("abc") == "unknown"
    assert xray_module._short_sha("a" * 64) == "a" * 12


def test_cli_list_plots_exits_zero(xray_module):
    rc = xray_module.main(["--list-plots"])
    assert rc == 0


def test_cli_rejects_tmp_output(xray_module):
    with pytest.raises(SystemExit, match="FORBIDDEN"):
        xray_module.main(["--archive-sha", "abc", "--output", "/tmp/should_fail.png"])


def test_canonical_plots_constant_pinned(xray_module):
    """Per Catalog #229 — pin the canonical 5-plot taxonomy."""
    assert "per_pair_distribution" in xray_module.CANONICAL_PLOTS
    assert "per_byte_heatmap" in xray_module.CANONICAL_PLOTS
    assert "cumulative_by_rank" in xray_module.CANONICAL_PLOTS
    assert "cross_substrate_correlation" in xray_module.CANONICAL_PLOTS
    assert "wyner_ziv_flow" in xray_module.CANONICAL_PLOTS
    assert "all" in xray_module.CANONICAL_PLOTS
    # Exactly 5 distinct plot types + "all" sentinel
    assert len(xray_module.CANONICAL_PLOTS) == 6


def test_axis_labels_pinned(xray_module):
    """Per CLAUDE.md axis-discipline — canonical (seg, pose, rate) order."""
    assert xray_module.AXIS_LABELS == ("seg", "pose", "rate")


def test_cli_missing_output_raises(xray_module):
    with pytest.raises(SystemExit, match="--output is required"):
        xray_module.main(["--archive-sha", "abc"])


def test_cli_missing_archive_sha_raises(xray_module, tmp_path):
    with pytest.raises(SystemExit, match="--archive-sha is required"):
        xray_module.main(["--output", str(tmp_path / "out.png")])
