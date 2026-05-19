# SPDX-License-Identifier: MIT
"""Tests for slot 10 xray --grain CLI + cascade_smearing_comparison plot.

Per Catalog #318 + codex op7 finding 2026-05-19. Slot 6 + slot 10
grain-awareness wave.
"""
from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


@pytest.fixture(scope="module")
def xray_module():
    """Import tools/master_gradient_xray.py as a module."""
    tools_dir = REPO_ROOT / "tools"
    sys.path.insert(0, str(tools_dir))
    try:
        return importlib.import_module("master_gradient_xray")
    finally:
        if str(tools_dir) in sys.path:
            sys.path.remove(str(tools_dir))


# ─────────────────────────────────────────────────────────────────────────
# Constants + CLI shape
# ─────────────────────────────────────────────────────────────────────────


def test_grain_filter_choices_canonical(xray_module) -> None:
    """4 canonical grain-filter values: raw_byte / post_decompress / compare_both / all."""
    assert xray_module.GRAIN_FILTER_RAW_BYTE == "raw_byte"
    assert xray_module.GRAIN_FILTER_POST_DECOMPRESS == "post_decompress"
    assert xray_module.GRAIN_FILTER_COMPARE_BOTH == "compare_both"
    assert xray_module.GRAIN_FILTER_ALL == "all"
    assert set(xray_module.GRAIN_FILTER_CHOICES) == {
        "raw_byte", "post_decompress", "compare_both", "all"
    }


def test_canonical_plots_includes_cascade_smearing_comparison(xray_module) -> None:
    """The 7th plot type is in the CANONICAL_PLOTS tuple."""
    assert "cascade_smearing_comparison" in xray_module.CANONICAL_PLOTS


def test_sidecar_schema_bumped_to_v2(xray_module) -> None:
    """Schema version bumped to v2 for grain-aware sister JSON contract."""
    assert xray_module.PLOT_SIDECAR_SCHEMA_VERSION == (
        "master_gradient_xray_plot_sidecar_v2_20260519"
    )
    assert xray_module.INDEX_HTML_SCHEMA_VERSION == (
        "master_gradient_xray_index_v2_20260519"
    )


def test_cli_help_advertises_grain_flag() -> None:
    """`--help` lists --grain with the 4 canonical choices."""
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "tools" / "master_gradient_xray.py"), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--grain" in result.stdout
    assert "raw_byte" in result.stdout
    assert "post_decompress" in result.stdout
    assert "compare_both" in result.stdout


def test_cli_help_advertises_cascade_smearing_plot() -> None:
    """`--help` lists cascade_smearing_comparison in --plot choices."""
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "tools" / "master_gradient_xray.py"), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "cascade_smearing_comparison" in result.stdout


def test_cli_list_plots_emits_cascade_smearing(xray_module) -> None:
    """`--list-plots` JSON includes cascade_smearing_comparison."""
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "master_gradient_xray.py"),
            "--list-plots",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    data = json.loads(result.stdout)
    assert "cascade_smearing_comparison" in data["canonical_plots"]


def test_cli_rejects_invalid_grain_value() -> None:
    """argparse rejects --grain values outside the canonical 4."""
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "master_gradient_xray.py"),
            "--grain", "invalid_grain",
            "--archive-sha", "a" * 64,
            "--output-dir", "/dev/null",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "invalid choice" in result.stderr.lower() or "invalid choice" in result.stdout.lower()


# ─────────────────────────────────────────────────────────────────────────
# Cascade-smearing metrics math
# ─────────────────────────────────────────────────────────────────────────


def test_cascade_smearing_metrics_identical_grains_jaccard_one(xray_module) -> None:
    """Identical gradients → Jaccard=1.0 → cascade_smearing_factor=0.0 → LOW."""
    arr = np.random.RandomState(42).randn(100, 3).astype(np.float32)
    metrics = xray_module._compute_cascade_smearing_metrics(arr, arr.copy(), top_k=20)
    assert metrics["top_k_jaccard"] == 1.0
    assert metrics["cascade_smearing_factor"] == 0.0
    assert metrics["verdict"] == "LOW"


def test_cascade_smearing_metrics_disjoint_top_k_high_verdict(xray_module) -> None:
    """Disjoint top-K → Jaccard near 0 → cascade_smearing_factor near 1 → HIGH."""
    n = 100
    raw = np.zeros((n, 3), dtype=np.float32)
    post = np.zeros((n, 3), dtype=np.float32)
    # Raw: top-K=10 are indices 0..9
    for i in range(10):
        raw[i, 0] = 10.0 - i * 0.1
    # Post: top-K=10 are indices 50..59 (disjoint)
    for i in range(10):
        post[50 + i, 0] = 10.0 - i * 0.1
    metrics = xray_module._compute_cascade_smearing_metrics(raw, post, top_k=10)
    assert metrics["top_k_jaccard"] == 0.0
    assert metrics["cascade_smearing_factor"] == 1.0
    assert metrics["verdict"] == "HIGH"


def test_cascade_smearing_metrics_partial_overlap_medium(xray_module) -> None:
    """Half overlap → Jaccard=1/3 → cascade_smearing_factor=2/3 → MEDIUM."""
    n = 100
    raw = np.zeros((n, 3), dtype=np.float32)
    post = np.zeros((n, 3), dtype=np.float32)
    # Raw top-10 = 0..9; Post top-10 = 5..14 → intersection=5, union=15, J=1/3
    for i in range(10):
        raw[i, 0] = 10.0 - i * 0.1
        post[5 + i, 0] = 10.0 - i * 0.1
    metrics = xray_module._compute_cascade_smearing_metrics(raw, post, top_k=10)
    assert metrics["top_k_jaccard"] == pytest.approx(1.0 / 3.0, abs=1e-3)
    assert metrics["verdict"] == "MEDIUM"


def test_cascade_smearing_metrics_handles_different_n_bytes(xray_module) -> None:
    """Different N_bytes (raw vs post-decompress) does not crash."""
    raw = np.random.RandomState(0).randn(50, 3).astype(np.float32)
    post = np.random.RandomState(1).randn(100, 3).astype(np.float32)
    metrics = xray_module._compute_cascade_smearing_metrics(raw, post, top_k=10)
    assert metrics["n_bytes_raw"] == 50
    assert metrics["n_bytes_post"] == 100
    assert "verdict" in metrics


def test_cascade_smearing_metrics_zero_bytes_returns_unknown(xray_module) -> None:
    """Empty grade arrays → UNKNOWN_INSUFFICIENT_DATA."""
    raw = np.zeros((0, 3), dtype=np.float32)
    post = np.zeros((0, 3), dtype=np.float32)
    metrics = xray_module._compute_cascade_smearing_metrics(raw, post, top_k=10)
    assert metrics["verdict"] == "UNKNOWN_INSUFFICIENT_DATA"


def test_cascade_smearing_metrics_spearman_perfectly_correlated(xray_module) -> None:
    """When magnitudes have identical rank order, Spearman ≈ 1.0."""
    arr = np.arange(100).reshape(100, 1).astype(np.float32)
    arr = np.hstack([arr, arr, arr])
    metrics = xray_module._compute_cascade_smearing_metrics(arr, arr.copy(), top_k=20)
    assert metrics["rank_correlation_spearman"] == pytest.approx(1.0, abs=1e-3)


# ─────────────────────────────────────────────────────────────────────────
# Cascade comparison plot end-to-end (synthetic)
# ─────────────────────────────────────────────────────────────────────────


def test_plot_cascade_smearing_writes_png_and_returns_metrics(
    xray_module, tmp_path: Path
) -> None:
    """plot_cascade_smearing_comparison writes a .png + returns metrics dict."""
    raw = np.random.RandomState(0).randn(50, 3).astype(np.float32)
    post = np.random.RandomState(1).randn(80, 3).astype(np.float32)
    raw_anchor = {
        "archive_sha256": "a" * 64,
        "gradient_byte_domain": "scored_archive_bytes",
        "measurement_axis": "[contest-CPU]",
        "measurement_hardware": "linux_x86_64_t4",
        "n_bytes": 50,
    }
    post_anchor = {
        "archive_sha256": "a" * 64,
        "gradient_byte_domain": "post_brotli_decompress_decoder_weight_bytes",
        "measurement_axis": "[contest-CPU]",
        "measurement_hardware": "linux_x86_64_t4",
        "n_bytes": 80,
    }
    out = tmp_path / "cascade.png"
    metrics = xray_module.plot_cascade_smearing_comparison(
        raw, raw_anchor, post, post_anchor, out, top_k=20
    )
    assert out.is_file()
    assert out.stat().st_size > 1000  # Non-empty PNG.
    assert "top_k_jaccard" in metrics
    assert "verdict" in metrics


# ─────────────────────────────────────────────────────────────────────────
# Live-repo regression guard
# ─────────────────────────────────────────────────────────────────────────


def test_live_xray_help_includes_grain_flag() -> None:
    """Live-repo regression: master_gradient_xray.py CLI carries --grain."""
    p = REPO_ROOT / "tools" / "master_gradient_xray.py"
    src = p.read_text()
    assert "--grain" in src
    assert "GRAIN_FILTER_CHOICES" in src
    assert "cascade_smearing_comparison" in src


def test_live_xray_includes_cascade_plot_helper() -> None:
    p = REPO_ROOT / "tools" / "master_gradient_xray.py"
    src = p.read_text()
    assert "plot_cascade_smearing_comparison" in src
    assert "_compute_cascade_smearing_metrics" in src
    assert "_load_aggregate_by_grain" in src


def test_live_grain_compare_both_smoke_runs_on_fec6(tmp_path: Path) -> None:
    """Live FEC6 archive: --grain compare_both runs without crashing.

    The FEC6 archive currently has ONLY raw-byte anchors so the cascade
    comparison plot is NOT emitted (per-archive eligibility check); the
    canonical 4 of 5 plots that don't require per-pair gradient should
    still emit successfully (per_pair_distribution skips because there
    is no per-pair anchor in the live ledger).
    """
    out_dir = tmp_path / "xray_smoke"
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "master_gradient_xray.py"),
            "--archive-sha",
            "6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf",
            "--grain", "compare_both",
            "--output-dir", str(out_dir),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    assert (out_dir / "index.html").exists()
    # Per_byte_heatmap should always emit when an anchor exists for the archive.
    assert (out_dir / "per_byte_heatmap.png").exists()


def test_live_grain_compare_both_index_html_carries_grain_banner(
    tmp_path: Path,
) -> None:
    """Index HTML carries the grain-awareness banner + grain-inventory column."""
    out_dir = tmp_path / "xray_grain_banner"
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "master_gradient_xray.py"),
            "--archive-sha",
            "6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf",
            "--grain", "compare_both",
            "--output-dir", str(out_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    html = (out_dir / "index.html").read_text()
    assert "Grain awareness" in html
    assert "Catalog #318" in html
    assert "grain inventory" in html.lower()
    assert "grain filter" in html.lower()
