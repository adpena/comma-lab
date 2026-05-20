# SPDX-License-Identifier: MIT
"""Tests for tools/probe_null_byte_master_gradient.py.

Covers helper unit (CLI flag parsing + synthetic anchor with known null
indices + epsilon sensitivity + per-section breakdown correctness) +
live-repo regression (the fec6 frontier OP3-V3 anchor produces 16292
null bytes at epsilon 1e-9 across the FP11/FEC6 grammar sections).
"""
from __future__ import annotations

import json
import struct
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

from tools.probe_null_byte_master_gradient import (
    parse_fec_grammar_from_inner_bytes,
    probe_null_bytes,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _synthetic_grad_with_known_nulls(
    n_bytes: int = 100,
    null_indices: tuple[int, ...] = (0, 5, 17, 42, 99),
) -> np.ndarray:
    """Return shape (n_bytes, 3) with zeros only at null_indices."""
    rng = np.random.default_rng(0)
    grad = rng.standard_normal((n_bytes, 3)).astype(np.float64) * 1e-3
    grad[list(null_indices), :] = 0.0
    return grad


def test_probe_synthetic_aggregate_known_null_indices() -> None:
    grad = _synthetic_grad_with_known_nulls()
    out = probe_null_bytes(grad=grad, epsilon=1e-9)
    assert out["n_total_bytes"] == 100
    assert out["n_null_bytes"] == 5
    assert out["null_fraction"] == 0.05
    np.testing.assert_array_equal(out["null_indices"], np.array([0, 5, 17, 42, 99]))


def test_probe_epsilon_threshold_sensitivity() -> None:
    """Epsilon controls inclusion: small nonzero values below epsilon are null."""
    grad = np.zeros((10, 3), dtype=np.float64)
    grad[3, 0] = 1e-12
    grad[7, 1] = 1e-7
    # At epsilon 1e-9: both 1e-12 and 1e-7? No: 1e-7 > 1e-9, so only the 1e-12 byte
    # plus all the all-zero bytes count.
    out_strict = probe_null_bytes(grad=grad, epsilon=1e-9)
    # bytes 0..9, all zero except [3]=1e-12 (below) and [7]=1e-7 (above)
    # So nulls = {0,1,2,3,4,5,6,8,9} = 9 indices
    assert out_strict["n_null_bytes"] == 9
    assert 7 not in out_strict["null_indices"].tolist()
    assert 3 in out_strict["null_indices"].tolist()
    out_loose = probe_null_bytes(grad=grad, epsilon=1.0)
    # epsilon=1.0 makes everything null
    assert out_loose["n_null_bytes"] == 10


def test_probe_per_pair_3d_collapses_via_max() -> None:
    """3D per-pair tensor (N_bytes, N_pairs, 3) must collapse via max-over-pairs."""
    grad = np.zeros((10, 4, 3), dtype=np.float64)
    # Byte 5: nonzero only on pair 2 → should NOT be null
    grad[5, 2, 0] = 0.1
    out = probe_null_bytes(grad=grad, epsilon=1e-9)
    assert out["n_total_bytes"] == 10
    assert out["n_null_bytes"] == 9
    assert 5 not in out["null_indices"].tolist()


def test_probe_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="must be 2D or 3D"):
        probe_null_bytes(grad=np.zeros(10), epsilon=1e-9)
    with pytest.raises(ValueError, match="must have shape"):
        probe_null_bytes(grad=np.zeros((10, 2)), epsilon=1e-9)


def _build_fake_fec6_inner(source_payload: bytes, selector_payload: bytes) -> bytes:
    """Build a minimal valid FP11/FEC6 wrapped inner-member bytes."""
    return (
        b"FP11"
        + struct.pack("<I", len(source_payload))
        + source_payload
        + struct.pack("<H", len(selector_payload))
        + selector_payload
    )


def test_grammar_parse_canonical_fp11_fec6() -> None:
    source = b"\x01\x02\x03" * 20  # 60 bytes
    selector = b"FEC6" + b"\xff" * 16  # 20 bytes
    inner = _build_fake_fec6_inner(source, selector)
    sections = parse_fec_grammar_from_inner_bytes(inner)
    assert sections is not None
    assert sections.outer_magic_end == 4
    assert sections.source_payload_end - sections.source_payload_start == 60
    assert sections.selector_payload_end - sections.selector_payload_start == 20
    assert sections.selector_magic == "FEC6"
    assert sections.total_bytes == len(inner)


def test_grammar_parse_rejects_non_fp11() -> None:
    sections = parse_fec_grammar_from_inner_bytes(b"WRONG" + b"\x00" * 100)
    assert sections is None
    # Truncated
    sections = parse_fec_grammar_from_inner_bytes(b"FP11")
    assert sections is None


def test_probe_per_section_breakdown_fec6() -> None:
    source = b"\x01\x02\x03" * 30  # 90 bytes (indices 8..98)
    selector = b"FEC6" + b"\xff" * 6  # 10 bytes (indices 100..110)
    inner = _build_fake_fec6_inner(source, selector)
    # Inner len = 4 + 4 + 90 + 2 + 10 = 110 bytes
    # Build a grad tensor that has nulls at: ALL headers + ALL selector + 5 source bytes
    n = 110
    grad = np.full((n, 3), 1e-3, dtype=np.float64)
    grad[0:4, :] = 0.0  # OUTER_MAGIC
    grad[4:8, :] = 0.0  # source_len_hdr
    grad[8:13, :] = 0.0  # 5 source bytes
    grad[98:100, :] = 0.0  # selector_len_hdr
    grad[100:110, :] = 0.0  # selector_payload
    out = probe_null_bytes(grad=grad, epsilon=1e-9, inner_bytes=inner)
    assert out["section_breakdown"]["OUTER_MAGIC"]["n_null"] == 4
    assert out["section_breakdown"]["source_len_hdr"]["n_null"] == 4
    assert out["section_breakdown"]["source_payload"]["n_null"] == 5
    assert out["section_breakdown"]["selector_len_hdr"]["n_null"] == 2
    assert out["section_breakdown"]["selector_payload"]["n_null"] == 10
    assert out["n_null_bytes"] == 4 + 4 + 5 + 2 + 10


def test_cli_end_to_end_with_synthetic_inputs(tmp_path: Path) -> None:
    """Run the CLI as a subprocess against synthetic anchor + archive.zip."""
    # Synthetic anchor
    grad = _synthetic_grad_with_known_nulls(n_bytes=50, null_indices=(0, 10, 20, 30, 40))
    anchor_path = tmp_path / "synthetic_anchor.npy"
    np.save(anchor_path, grad)
    # Synthetic archive.zip with one inner member; total inner = 4 (FP11) + 4 (src_len)
    # + N_source + 2 (sel_len) + N_selector = 10 + N_source + N_selector. For N_total=50
    # with N_selector=10 (FEC6 + 6 trailer bytes), N_source = 50 - 10 - 10 = 30.
    inner = _build_fake_fec6_inner(b"\x01" * 30, b"FEC6" + b"\xff" * 6)
    assert len(inner) == 50, f"inner len = {len(inner)}"
    archive_path = tmp_path / "synthetic_archive.zip"
    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.writestr("x", inner)
    output_dir = tmp_path / "probe_out"
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "probe_null_byte_master_gradient.py"),
            "--anchor",
            str(anchor_path),
            "--archive-zip",
            str(archive_path),
            "--inner-member-name",
            "x",
            "--epsilon",
            "1e-9",
            "--output-dir",
            str(output_dir),
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, f"CLI failed: stderr={result.stderr}"
    summary = json.loads((output_dir / "null_byte_summary.json").read_text())
    assert summary["n_null_bytes"] == 5
    assert summary["n_total_bytes"] == 50
    assert summary["score_claim"] is False
    assert summary["promotable"] is False
    assert summary["axis_tag"] == "[predicted]"
    assert summary["schema"] == "null_byte_master_gradient_probe_v1"
    assert summary["provenance"]["artifact_kind"] == "PREDICTED_FROM_MODEL"
    indices = np.load(output_dir / "null_byte_indices.npy")
    np.testing.assert_array_equal(indices, np.array([0, 10, 20, 30, 40]))


def test_live_op3v3_fec6_frontier_anchor_regression() -> None:
    """Regression: the canonical OP3-V3 anchor produces exactly 16292 null bytes."""
    anchor_path = REPO_ROOT / ".omx/state/master_gradient_fec6_contest_cuda_t4_20260520.npy"
    if not anchor_path.is_file():
        pytest.skip("OP3-V3 anchor not present on this checkout (CI / fresh clone)")
    grad = np.load(anchor_path)
    archive_path = (
        REPO_ROOT
        / "experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/archive.zip"
    )
    if archive_path.is_file():
        with zipfile.ZipFile(archive_path) as zf:
            with zf.open("x") as f:
                inner = f.read()
    else:
        inner = None
    out = probe_null_bytes(grad=grad, epsilon=1e-9, inner_bytes=inner)
    assert out["n_total_bytes"] == 178417
    assert out["n_null_bytes"] == 16292
    assert abs(out["null_fraction"] - 0.09131416849291267) < 1e-9
    if inner is not None:
        assert out["section_breakdown"]["selector_payload"]["null_fraction_within_section"] == 1.0
        assert out["section_breakdown"]["OUTER_MAGIC"]["null_fraction_within_section"] == 1.0
