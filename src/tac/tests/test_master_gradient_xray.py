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
    """Per Catalog #229 — pin the canonical 5-plot taxonomy + optional drift plot."""
    assert "per_pair_distribution" in xray_module.CANONICAL_PLOTS
    assert "per_byte_heatmap" in xray_module.CANONICAL_PLOTS
    assert "cumulative_by_rank" in xray_module.CANONICAL_PLOTS
    assert "cross_substrate_correlation" in xray_module.CANONICAL_PLOTS
    assert "wyner_ziv_flow" in xray_module.CANONICAL_PLOTS
    assert "drift_vs_sensitivity_scatter" in xray_module.CANONICAL_PLOTS
    # Slot 10 grain-awareness wave 2026-05-19: 7th plot type
    assert "cascade_smearing_comparison" in xray_module.CANONICAL_PLOTS
    # Slot EE 2026-05-19 task #797 extensions: 2 new plot types
    assert "consumer_verdict_matrix" in xray_module.CANONICAL_PLOTS
    assert "provenance_audit_timeline" in xray_module.CANONICAL_PLOTS
    assert "all" in xray_module.CANONICAL_PLOTS
    # 5 canonical + 1 optional drift + 1 cascade + 2 slot-EE + "all" sentinel = 10
    assert len(xray_module.CANONICAL_PLOTS) == 10
    # CANONICAL_OUTPUT_DIR_PLOTS is the canonical 5 (no drift, no all)
    assert xray_module.CANONICAL_OUTPUT_DIR_PLOTS == (
        "per_pair_distribution",
        "per_byte_heatmap",
        "cumulative_by_rank",
        "cross_substrate_correlation",
        "wyner_ziv_flow",
    )


def test_axis_labels_pinned(xray_module):
    """Per CLAUDE.md axis-discipline — canonical (seg, pose, rate) order."""
    assert xray_module.AXIS_LABELS == ("seg", "pose", "rate")


def test_cli_missing_output_raises(xray_module):
    with pytest.raises(
        SystemExit, match=r"--output \(single plot file\) OR --output-dir"
    ):
        xray_module.main(["--archive-sha", "abc"])


def test_cli_output_and_output_dir_mutually_exclusive(xray_module, tmp_path):
    with pytest.raises(SystemExit, match="mutually exclusive"):
        xray_module.main(
            [
                "--archive-sha",
                "abc",
                "--output",
                str(tmp_path / "plot.png"),
                "--output-dir",
                str(tmp_path / "dir"),
            ]
        )


def test_cli_rejects_tmp_output_dir(xray_module):
    with pytest.raises(SystemExit, match="FORBIDDEN"):
        xray_module.main(
            ["--archive-sha", "abc", "--output-dir", "/tmp/should_fail/"]
        )


# ──────────────────────────────────────────────────────────────────────────── #
# New: Plot 6 — drift_vs_sensitivity_scatter                                   #
# ──────────────────────────────────────────────────────────────────────────── #


@pytest.fixture
def mps_drift_data():
    """Synthetic MPS drift JSON payload mirroring the canonical schema."""
    return {
        "schema_version": "mps_drift_granular_v1_test",
        "axis_tag": "[macOS-MPS-PyTorch-vs-CUDA-diagnostic]",
        "evidence_grade": "MPS-research-signal",
        "n_pairs": 8,
        "per_pair": [
            {"pair_index": i, "aggregate": 0.1 + i * 0.01} for i in range(8)
        ],
        "per_frame": [
            {"pair_index": i, "frame_index": 0, "aggregate": 0.3 + i * 0.005}
            for i in range(8)
        ],
        "score_claim": False,
        "promotion_eligible": False,
    }


def test_plot_drift_vs_sensitivity_scatter_with_per_pair(
    xray_module, synthetic_per_pair, synthetic_aggregate, anchor_basic,
    mps_drift_data, tmp_path,
):
    out = tmp_path / "drift_scatter.png"
    xray_module.plot_drift_vs_sensitivity_scatter(
        synthetic_per_pair, synthetic_aggregate, anchor_basic, mps_drift_data, out
    )
    assert out.exists()
    assert out.stat().st_size > 1000


def test_plot_drift_vs_sensitivity_scatter_per_frame_fallback(
    xray_module, synthetic_aggregate, anchor_basic, mps_drift_data, tmp_path,
):
    """When per-pair gradient is None, plot falls back to per-frame drift."""
    out = tmp_path / "drift_scatter_per_frame.png"
    xray_module.plot_drift_vs_sensitivity_scatter(
        None, synthetic_aggregate, anchor_basic, mps_drift_data, out
    )
    assert out.exists()


def test_plot_drift_vs_sensitivity_scatter_empty_drift_renders_placeholder(
    xray_module, synthetic_aggregate, anchor_basic, tmp_path,
):
    """Empty drift JSON should render a placeholder PNG, not crash."""
    out = tmp_path / "drift_empty.png"
    xray_module.plot_drift_vs_sensitivity_scatter(
        None,
        synthetic_aggregate,
        anchor_basic,
        {"per_pair": [], "per_frame": []},
        out,
    )
    assert out.exists()


def test_load_mps_drift_json_missing_file_raises(xray_module, tmp_path):
    with pytest.raises(SystemExit, match="does not exist"):
        xray_module._load_mps_drift_json(tmp_path / "missing.json")


def test_load_mps_drift_json_malformed_raises(xray_module, tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("not json {")
    with pytest.raises(SystemExit, match="failed to parse"):
        xray_module._load_mps_drift_json(bad)


# ──────────────────────────────────────────────────────────────────────────── #
# New: sister JSON sidecar emission + canonical Provenance per Catalog #323    #
# ──────────────────────────────────────────────────────────────────────────── #


def test_summary_stats_for_aggregate_pinned_schema(xray_module, synthetic_aggregate):
    summary = xray_module._summary_stats_for_aggregate(synthetic_aggregate)
    required_keys = {
        "n_bytes",
        "n_axes",
        "per_axis_l1_total",
        "per_axis_max",
        "per_byte_l1_max",
        "per_byte_l1_median",
        "per_byte_l1_total",
        "top_1pct_leverage_fraction",
        "top_10pct_leverage_fraction",
        "top_10_byte_indices",
    }
    assert required_keys.issubset(summary.keys())
    assert summary["n_bytes"] == 128
    assert summary["n_axes"] == 3
    assert len(summary["top_10_byte_indices"]) == 10


def test_summary_stats_for_per_pair_pinned_schema(xray_module, synthetic_per_pair):
    summary = xray_module._summary_stats_for_per_pair(synthetic_per_pair)
    required_keys = {
        "n_bytes",
        "n_pairs",
        "n_axes",
        "per_pair_l1_mean",
        "per_pair_l1_median",
        "per_pair_l1_max",
        "per_pair_per_axis_max",
        "top_10_pair_indices_by_l1",
    }
    assert required_keys.issubset(summary.keys())
    assert summary["n_pairs"] == 12
    assert len(summary["top_10_pair_indices_by_l1"]) == 10


def test_emit_sidecar_json_carries_canonical_provenance(
    xray_module, anchor_basic, synthetic_aggregate, tmp_path,
):
    """Per Catalog #323: sister JSON MUST carry canonical Provenance sub-object."""
    sidecar = tmp_path / "test_plot.json"
    summary = xray_module._summary_stats_for_aggregate(synthetic_aggregate)
    xray_module._emit_sidecar_json(
        sidecar,
        plot_id="test_plot",
        anchor=anchor_basic,
        summary=summary,
        extra={"unit_test": True},
    )
    assert sidecar.exists()
    import json

    payload = json.loads(sidecar.read_text())
    # Canonical Provenance sub-object per Catalog #323
    assert "provenance" in payload
    prov = payload["provenance"]
    assert prov["artifact_kind"] == "predicted_from_model"
    assert prov["evidence_grade"] == "predicted"
    assert prov["promotion_eligible"] is False
    assert prov["score_claim_valid"] is False
    assert prov["measurement_axis"] == "[predicted]"
    assert (
        prov["canonical_helper_invocation"]
        == "tac.provenance.builders.build_provenance_for_predicted"
    )
    # Catalog #287 + #323: top-level fail-closed defaults
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
    # Schema pinned
    assert payload["schema_version"] == xray_module.PLOT_SIDECAR_SCHEMA_VERSION
    assert payload["plot_id"] == "test_plot"
    assert payload["extra"] == {"unit_test": True}
    # Anchor fields preserved for downstream consumers
    anchor_dump = payload["anchor"]
    assert anchor_dump["archive_sha256"] == anchor_basic["archive_sha256"]
    assert anchor_dump["measurement_axis"] == anchor_basic["measurement_axis"]


def test_sha256_anchor_fingerprint_is_deterministic(xray_module, anchor_basic):
    fp1 = xray_module._sha256_anchor_fingerprint(anchor_basic)
    fp2 = xray_module._sha256_anchor_fingerprint(anchor_basic)
    assert fp1 == fp2
    assert len(fp1) == 64  # sha256 hex


def test_sha256_anchor_fingerprint_differs_when_inputs_differ(
    xray_module, anchor_basic, anchor_with_sections,
):
    fp_basic = xray_module._sha256_anchor_fingerprint(anchor_basic)
    fp_sections = xray_module._sha256_anchor_fingerprint(anchor_with_sections)
    assert fp_basic != fp_sections


# ──────────────────────────────────────────────────────────────────────────── #
# New: index.html schema + structure                                            #
# ──────────────────────────────────────────────────────────────────────────── #


def test_emit_index_html_landing_page_structure(
    xray_module, anchor_basic, anchor_with_sections, tmp_path,
):
    """Index HTML carries the canonical observability banner + plot grid."""
    out_dir = tmp_path / "xray_run"
    out_dir.mkdir()
    archive_shas = [anchor_basic["archive_sha256"], anchor_with_sections["archive_sha256"]]
    anchors = [anchor_basic, anchor_with_sections]
    plots = [
        {
            "plot_id": "per_byte_heatmap",
            "png_relative": "per_byte_heatmap.png",
            "sidecar_relative": "per_byte_heatmap.json",
        },
        {
            "plot_id": "cumulative_by_rank",
            "png_relative": "cumulative_by_rank.png",
            "sidecar_relative": "cumulative_by_rank.json",
        },
    ]
    xray_module._emit_index_html(
        out_dir, archive_shas, anchors, plots, mps_drift_json_path=None
    )
    index = out_dir / "index.html"
    assert index.exists()
    html = index.read_text(encoding="utf-8")
    # Banner per Catalog #305 + #323
    assert "Observability surface" in html
    assert "score_claim=false" in html.lower() or "score_claim=False" in html
    assert "evidence_grade=[predicted]" in html
    # Plot cards present
    assert "per_byte_heatmap.png" in html
    assert "per_byte_heatmap.json" in html
    # Archive table present
    assert "<table>" in html
    assert "n_bytes" in html
    # Schema version pinned
    assert xray_module.INDEX_HTML_SCHEMA_VERSION in html
    # Cross-references include canonical helpers
    assert "tac.master_gradient_consumers" in html
    assert "Catalog #305" in html
    assert "Catalog #323" in html


def test_emit_index_html_with_mps_drift_path(
    xray_module, anchor_basic, tmp_path,
):
    """When mps_drift_json_path is set, cross-ref list mentions it."""
    out_dir = tmp_path / "xray_drift_run"
    out_dir.mkdir()
    drift_file = tmp_path / "synthetic_drift.json"
    drift_file.write_text("{}")
    xray_module._emit_index_html(
        out_dir,
        [anchor_basic["archive_sha256"]],
        [anchor_basic],
        [],
        mps_drift_json_path=drift_file,
    )
    html = (out_dir / "index.html").read_text(encoding="utf-8")
    assert "MPS drift cross-reference" in html


# ──────────────────────────────────────────────────────────────────────────── #
# CLI: --list-plots includes new plot                                          #
# ──────────────────────────────────────────────────────────────────────────── #


def test_cli_list_plots_includes_drift_and_schema_versions(
    xray_module, capsys,
):
    rc = xray_module.main(["--list-plots"])
    assert rc == 0
    captured = capsys.readouterr().out
    import json as _json

    payload = _json.loads(captured)
    assert "drift_vs_sensitivity_scatter" in payload["canonical_plots"]
    assert "sidecar_schema_version" in payload
    assert "index_html_schema_version" in payload
    # Schema bumped to v2 by slot 10 grain-awareness wave 2026-05-19.
    assert payload["sidecar_schema_version"].startswith("master_gradient_xray_plot_sidecar_v2")


def test_cli_missing_archive_sha_raises(xray_module, tmp_path):
    with pytest.raises(SystemExit, match="--archive-sha is required"):
        xray_module.main(["--output", str(tmp_path / "out.png")])
