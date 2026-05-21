# SPDX-License-Identifier: MIT
"""Tests for tools.build_sensitivity_weighted_foveation_params_generator.

OVERNIGHT-X1 Builder 1 of 2. Covers:
 - schema invariant: output is List[FoveationParamsRow] of exact length n_frames_out
 - byte-stable round-trip: sha256 deterministic for same input
 - Catalog #318 chain-rule respect: typed ContestGradientTensor input shape
   contract (4-D (N_pairs, 3, H, W)); rejection on raw byte arrays
 - smoke mode runs without paid GPU (no Modal/Vast invocations; CPU-only)
 - graceful failure on missing input
 - output schema matches HFV1 foveation_params.bin grammar
"""
from __future__ import annotations

import json
import struct
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("numpy")
import numpy as np

# Import the builder module.
from tools import build_sensitivity_weighted_foveation_params_generator as gen


REPO_ROOT = Path(__file__).resolve().parents[3]
BUILDER_PATH = REPO_ROOT / "tools" / "build_sensitivity_weighted_foveation_params_generator.py"


# -----------------------------------------------------------------------------
# Test 1 - schema invariant
# -----------------------------------------------------------------------------


def test_canonical_hfv1_constants_match_inflate_runtime():
    """HFV1_HEADER / HFV1_ROW / HFV1_MAGIC must match the inflate runtime contract."""
    assert gen.HFV1_MAGIC == b"HFV1"
    assert gen.HFV1_HEADER.size == 16  # "<4sIII" = 4 + 4 + 4 + 4
    assert gen.HFV1_ROW.size == 20  # "<fffff" = 5 * 4
    assert gen.CAMERA_H == 874
    assert gen.CAMERA_W == 1164


def test_schema_invariant_output_length_matches_n_frames_out():
    m = gen.build_synthetic_m_contest_fixture(n_pairs=4, height=16, width=16, seed=7)
    rows, obs = gen.generate_sensitivity_weighted_foveation_params(
        m, n_frames_out=8
    )
    assert len(rows) == 8
    assert len(obs) == 8
    for i, row in enumerate(rows):
        assert isinstance(row, gen.FoveationParamsRow)
        assert row.frame_index == i


def test_schema_n_frames_padding_emits_zero_rows():
    """When n_frames_out > 2*N_pairs, padding frames emit zero-row."""
    m = gen.build_synthetic_m_contest_fixture(n_pairs=2, height=8, width=8, seed=1)
    rows, _ = gen.generate_sensitivity_weighted_foveation_params(
        m, n_frames_out=8
    )
    # First 4 frames have sensitivity (2 pairs * 2 frames); next 4 are zero-rows.
    active = sum(1 for r in rows if abs(r.alpha) > 1e-6)
    zero_rows = sum(1 for r in rows if abs(r.alpha) <= 1e-6)
    assert active == 4
    assert zero_rows == 4


# -----------------------------------------------------------------------------
# Test 2 - byte-stable round-trip
# -----------------------------------------------------------------------------


def test_byte_stable_round_trip_pack_unpack():
    """Pack + unpack round-trips losslessly."""
    m = gen.build_synthetic_m_contest_fixture(n_pairs=4, height=16, width=16, seed=42)
    rows, _ = gen.generate_sensitivity_weighted_foveation_params(
        m, n_frames_out=8, output_frame_height=64, output_frame_width=64
    )
    payload = gen.pack_hfv1_foveation_params(
        rows, frame_height=64, frame_width=64
    )
    n_frames, h, w, decoded = gen.unpack_hfv1_foveation_params(payload)
    assert n_frames == 8
    assert h == 64
    assert w == 64
    assert len(decoded) == 8
    # Per-row alpha/radius/power/origin_x/origin_y match within float32 rounding.
    for original, parsed in zip(rows, decoded):
        assert abs(original.alpha - parsed.alpha) < 1e-5
        assert abs(original.radius - parsed.radius) < 1e-3
        assert abs(original.power - parsed.power) < 1e-5
        assert abs(original.origin_x - parsed.origin_x) < 1e-3
        assert abs(original.origin_y - parsed.origin_y) < 1e-3


def test_byte_stable_sha256_deterministic_for_deterministic_input():
    """Same input + same seed => identical output sha256."""
    m1 = gen.build_synthetic_m_contest_fixture(n_pairs=4, height=16, width=16, seed=99)
    m2 = gen.build_synthetic_m_contest_fixture(n_pairs=4, height=16, width=16, seed=99)
    assert np.array_equal(m1, m2)

    rows1, _ = gen.generate_sensitivity_weighted_foveation_params(m1, n_frames_out=8)
    rows2, _ = gen.generate_sensitivity_weighted_foveation_params(m2, n_frames_out=8)
    payload1 = gen.pack_hfv1_foveation_params(rows1)
    payload2 = gen.pack_hfv1_foveation_params(rows2)
    assert payload1 == payload2
    assert gen._bytes_sha256(payload1) == gen._bytes_sha256(payload2)


def test_different_seeds_yield_different_payloads():
    m1 = gen.build_synthetic_m_contest_fixture(n_pairs=4, height=16, width=16, seed=1)
    m2 = gen.build_synthetic_m_contest_fixture(n_pairs=4, height=16, width=16, seed=2)
    rows1, _ = gen.generate_sensitivity_weighted_foveation_params(m1, n_frames_out=8)
    rows2, _ = gen.generate_sensitivity_weighted_foveation_params(m2, n_frames_out=8)
    payload1 = gen.pack_hfv1_foveation_params(rows1)
    payload2 = gen.pack_hfv1_foveation_params(rows2)
    assert payload1 != payload2


# -----------------------------------------------------------------------------
# Test 3 - Catalog #318 chain-rule respect: typed input contract
# -----------------------------------------------------------------------------


def test_rejects_wrong_shape_3d_array():
    """Catalog #318: only typed (N_pairs, 3, H, W) accepted; raw 3D rejected."""
    bad = np.zeros((4, 16, 16), dtype=np.float64)  # 3D
    with pytest.raises(ValueError, match="shape"):
        gen.generate_sensitivity_weighted_foveation_params(bad, n_frames_out=8)


def test_rejects_wrong_channel_dim():
    """Catalog #318: channel axis MUST be 3 (seg/pose/rate canonical layout)."""
    bad = np.zeros((4, 5, 16, 16), dtype=np.float64)  # 5 channels
    with pytest.raises(ValueError, match="shape"):
        gen.generate_sensitivity_weighted_foveation_params(bad, n_frames_out=8)


def test_per_frame_kernel_rejects_non_2d_sensitivity_map():
    bad = np.zeros((3, 16, 16), dtype=np.float64)
    with pytest.raises(ValueError, match="2D"):
        gen.compute_per_frame_foveation_row(bad, frame_index=0)


# -----------------------------------------------------------------------------
# Test 4 - smoke mode runs without paid GPU
# -----------------------------------------------------------------------------


def test_smoke_mode_runs_cpu_only_no_gpu(tmp_path):
    """Smoke mode produces a valid bin output without any GPU/Modal call."""
    out_bin = tmp_path / "smoke_foveation_params.bin"
    out_report = tmp_path / "smoke_report.json"
    rc = gen.main(
        [
            "--smoke",
            "--output-foveation-params-bin",
            str(out_bin),
            "--report-out-json",
            str(out_report),
        ]
    )
    assert rc == 0
    assert out_bin.is_file()
    assert out_bin.stat().st_size > 16
    assert out_report.is_file()
    report = json.loads(out_report.read_text())
    assert report["schema"].startswith(
        "build_sensitivity_weighted_foveation_params_generator_v1"
    )
    # Catalog #287/#323 evidence-tag discipline.
    assert report["score_claim"] is False
    assert report["promotion_eligible"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    assert report["axis_tag"] == "[predicted]"
    assert report["evidence_grade"] == "predicted"
    # Catalog #318: provenance carries chain-rule respect signal.
    assert "provenance" in report
    assert report["provenance"].get("artifact_kind") in (
        "predicted_from_model",
        "PREDICTED_FROM_MODEL",
    )


def test_smoke_dry_run_does_not_write_files(tmp_path):
    out_bin = tmp_path / "should_not_exist.bin"
    rc = gen.main(
        ["--smoke", "--output-foveation-params-bin", str(out_bin), "--dry-run"]
    )
    assert rc == 0
    assert not out_bin.exists()


def test_smoke_pr101_full_scale_produces_correct_byte_layout(tmp_path):
    """At full scale (600 pairs * 5 floats per frame * 2 frames per pair), expect
    standard HFV1 byte layout: 16 header + N_frames * 20 = 24,016 bytes for 1200 frames."""
    # Use smaller smoke (synthetic) but emit at PR101 frame count.
    out_bin = tmp_path / "scale_test.bin"
    rc = gen.main(
        [
            "--smoke",
            "--smoke-n-pairs",
            "4",
            "--smoke-height",
            "32",
            "--smoke-width",
            "32",
            "--n-frames-out",
            "1200",
            "--output-foveation-params-bin",
            str(out_bin),
        ]
    )
    assert rc == 0
    raw = out_bin.read_bytes()
    expected_bytes = 16 + 1200 * 20  # 24,016 bytes
    assert len(raw) == expected_bytes
    # Parse the header to confirm magic + n_frames + dimensions.
    magic = raw[:4]
    n_frames = struct.unpack_from("<I", raw, 4)[0]
    height = struct.unpack_from("<I", raw, 8)[0]
    width = struct.unpack_from("<I", raw, 12)[0]
    assert magic == b"HFV1"
    assert n_frames == 1200
    assert height == gen.CAMERA_H
    assert width == gen.CAMERA_W


# -----------------------------------------------------------------------------
# Test 5 - graceful failure on missing input
# -----------------------------------------------------------------------------


def test_missing_master_gradient_input_returns_nonzero(tmp_path, capsys):
    out_bin = tmp_path / "missing.bin"
    rc = gen.main(
        [
            "--master-gradient-tensor-npy",
            str(tmp_path / "does_not_exist.npy"),
            "--output-foveation-params-bin",
            str(out_bin),
        ]
    )
    assert rc == 2
    err = capsys.readouterr().err
    assert "not found" in err.lower()
    assert not out_bin.exists()


def test_missing_master_gradient_input_without_smoke_returns_nonzero(tmp_path, capsys):
    out_bin = tmp_path / "missing.bin"
    rc = gen.main(["--output-foveation-params-bin", str(out_bin)])
    assert rc == 2
    err = capsys.readouterr().err
    assert "required" in err.lower()


# -----------------------------------------------------------------------------
# Test 6 - output schema matches HFV1 foveation_params.bin grammar
# -----------------------------------------------------------------------------


def test_output_schema_matches_existing_hfv1_foveation_params_bin_grammar(tmp_path):
    """The output bin MUST be inflate-compatible with HFV1 inflate runtime."""
    out_bin = tmp_path / "out.bin"
    rc = gen.main(
        [
            "--smoke",
            "--smoke-n-pairs",
            "2",
            "--smoke-height",
            "8",
            "--smoke-width",
            "8",
            "--n-frames-out",
            "4",
            "--output-foveation-params-bin",
            str(out_bin),
        ]
    )
    assert rc == 0
    raw = out_bin.read_bytes()
    # Decode using our canonical unpack.
    n_frames, h, w, rows = gen.unpack_hfv1_foveation_params(raw)
    assert n_frames == 4
    assert h == gen.CAMERA_H
    assert w == gen.CAMERA_W
    assert len(rows) == 4
    # All rows must have valid 5-tuple semantics.
    for row in rows:
        assert isinstance(row.alpha, float)
        assert isinstance(row.radius, float)
        assert isinstance(row.power, float)
        assert isinstance(row.origin_x, float)
        assert isinstance(row.origin_y, float)
        # Alpha within safe clamp band.
        assert -1e-6 <= row.alpha <= gen.DEFAULT_ALPHA_CLAMP_MAX + 1e-6
        # Radius non-negative.
        assert row.radius >= gen.DEFAULT_RADIUS_FLOOR - 1e-6
        # Origin within frame bounds.
        assert 0 <= row.origin_x <= float(w)
        assert 0 <= row.origin_y <= float(h)


def test_byte_layout_first_16_bytes_match_HFV1_header_struct():
    """First 16 bytes MUST be (4s magic + I n_frames + I height + I width)."""
    m = gen.build_synthetic_m_contest_fixture(n_pairs=2, height=8, width=8, seed=1)
    rows, _ = gen.generate_sensitivity_weighted_foveation_params(m, n_frames_out=4)
    payload = gen.pack_hfv1_foveation_params(rows, frame_height=874, frame_width=1164)
    assert payload[:4] == b"HFV1"
    n, h, w = struct.unpack_from("<III", payload, 4)
    assert n == 4
    assert h == 874
    assert w == 1164
    # Each subsequent 20-byte block is 5 float32 (HFV1_ROW).
    for i in range(4):
        offset = 16 + i * 20
        alpha, radius, power, origin_x, origin_y = struct.unpack_from(
            "<fffff", payload, offset
        )
        # Float32 reconstruction within tolerance.
        assert alpha == pytest.approx(rows[i].alpha, abs=1e-5)
        assert radius == pytest.approx(rows[i].radius, rel=1e-3, abs=1e-3)
        assert power == pytest.approx(rows[i].power, abs=1e-5)
        assert origin_x == pytest.approx(rows[i].origin_x, abs=1e-3)
        assert origin_y == pytest.approx(rows[i].origin_y, abs=1e-3)


# -----------------------------------------------------------------------------
# Test 7 - sensitivity-of-mass mapping correctness
# -----------------------------------------------------------------------------


def test_center_of_mass_at_synthetic_blob_center():
    """A Gaussian blob centered at (cx, cy) should produce origin near (cx, cy)."""
    H, W = 32, 32
    cx, cy = 20.0, 12.0
    sigma = 3.0
    yy = np.arange(H, dtype=np.float64).reshape(H, 1)
    xx = np.arange(W, dtype=np.float64).reshape(1, W)
    s = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma ** 2))
    row, obs = gen.compute_per_frame_foveation_row(s, frame_index=0)
    assert obs["is_degenerate"] is False
    # Center-of-mass should be within 0.5 pixel of the Gaussian peak.
    assert abs(row.origin_x - cx) < 0.5
    assert abs(row.origin_y - cy) < 0.5
    # Sigma should be within ~10% of the Gaussian sigma.
    assert obs["sigma_xy"][0] == pytest.approx(sigma, rel=0.15)
    assert obs["sigma_xy"][1] == pytest.approx(sigma, rel=0.15)


def test_degenerate_zero_sensitivity_emits_zero_row():
    H, W = 16, 16
    s = np.zeros((H, W), dtype=np.float64)
    row, obs = gen.compute_per_frame_foveation_row(s, frame_index=5)
    assert obs["is_degenerate"] is True
    assert row.alpha == 0.0
    assert row.origin_x == W / 2.0
    assert row.origin_y == H / 2.0


def test_alpha_clamp_band_respected():
    """Alpha output MUST be in [alpha_clamp_min, alpha_clamp_max]."""
    H, W = 16, 16
    # Extremely concentrated, high-magnitude sensitivity to push alpha large.
    s = np.zeros((H, W), dtype=np.float64)
    s[8, 8] = 1e9
    row, _ = gen.compute_per_frame_foveation_row(
        s, frame_index=0, alpha_clamp_min=0.0, alpha_clamp_max=0.5
    )
    assert 0.0 <= row.alpha <= 0.5 + 1e-9


# -----------------------------------------------------------------------------
# Test 8 - operator-routable vocabulary aliases
# -----------------------------------------------------------------------------


def test_foveation_row_carries_operator_routable_aliases():
    row = gen.FoveationParamsRow(
        frame_index=0,
        alpha=0.3,
        radius=5.0,
        power=1.0,
        origin_x=100.0,
        origin_y=200.0,
    )
    d = row.as_dict()
    # HFV1 canonical names
    assert d["alpha"] == 0.3
    assert d["radius"] == 5.0
    assert d["power"] == 1.0
    assert d["origin_x"] == 100.0
    assert d["origin_y"] == 200.0
    # Operator-routable foveation-grammar aliases
    assert d["fovx_centerframe"] == 100.0
    assert d["fovy_centerframe"] == 200.0
    assert d["fov_z"] == 1.0


# -----------------------------------------------------------------------------
# Test 9 - CLI subprocess integration
# -----------------------------------------------------------------------------


def test_cli_subprocess_smoke_invocation(tmp_path):
    """Verify the builder can be invoked as a subprocess with --smoke."""
    out_bin = tmp_path / "subprocess_smoke.bin"
    out_report = tmp_path / "subprocess_report.json"
    result = subprocess.run(
        [
            sys.executable,
            str(BUILDER_PATH),
            "--smoke",
            "--output-foveation-params-bin",
            str(out_bin),
            "--report-out-json",
            str(out_report),
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, f"stderr={result.stderr}"
    assert out_bin.is_file()
    assert out_report.is_file()
    assert "WROTE" in result.stdout


# -----------------------------------------------------------------------------
# Test 10 - canonical equation references per Catalog #344
# -----------------------------------------------------------------------------


def test_canonical_equation_references_present():
    assert gen.CANONICAL_EQUATION_NAME == (
        "procedural_predictor_plus_residual_correction_savings_v1"
    )
    assert gen.CANONICAL_EQUATION_IN_DOMAIN_CONTEXT == (
        "hfv1_foveation_params_sensitivity_weighted_v1"
    )


def test_provenance_carries_canonical_helper_invocation():
    prov = gen._build_provenance(
        m_contest_sha="abc123",
        n_pairs=4,
        n_frames=8,
        height=16,
        width=16,
    )
    # Either canonical Provenance (from tac.provenance.builders) or our fallback;
    # both MUST carry the canonical model id signal somewhere queryable.
    # Canonical builder embeds model_id in `source_path` with a <predictor:...>
    # token; our fallback embeds it as a literal `model_id` field.
    source_path = prov.get("source_path", "")
    helper_invocation = prov.get("canonical_helper_invocation", "")
    assert (
        prov.get("model_id") == gen._PROVENANCE_MODEL_ID
        or gen._PROVENANCE_MODEL_ID in source_path
        or helper_invocation.startswith("tools/")
        or helper_invocation.startswith("tac.provenance.")
    )
    # Catalog #287/#323 evidence-grade contract: ALWAYS non-promotable predicted.
    assert prov.get("evidence_grade") == "predicted"
    assert prov.get("promotion_eligible") is False
    assert prov.get("score_claim_valid") is False
    assert prov.get("measurement_axis") == "[predicted]"
    assert prov.get("artifact_kind") in (
        "predicted_from_model",
        "PREDICTED_FROM_MODEL",
    )
