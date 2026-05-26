# SPDX-License-Identifier: MIT
"""Tests for the sister Catalog #1265 gate parameterized for Z6PCWM1 grammar.

Covers:
- Live-repo regression guard on the canonical D=Z6 L1 promotion archive
- Raw Z6PCWM1 bytes input path
- Zipped ``0.bin`` member input path
- Synthetic drift FAIL path (state_dict perturbation flips PASS → FAIL)
- Canonical Provenance + non-promotable marker assertions per Catalog
  #287/#323/#192/#1/#317/#341
- Invalid input rejection (bad magic / missing ZIP member / non-existent file)
- CLI exit-code contract: 0 = PASS / 1 = FAIL / 2 = error
"""

from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
GATE_TOOL = REPO_ROOT / "tools" / "gate_mlx_candidate_contest_equivalence_z6.py"
CANONICAL_Z6_L1_ARCHIVE = (
    REPO_ROOT / ".omx" / "tmp" / "z6_mlx_l1_converge_smoke" / "0.bin"
)
PY = sys.executable


def _has_mlx() -> bool:
    try:
        import mlx.core  # noqa: F401

        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_mlx(), reason="mlx not available")


def _load_module():
    """Import the canonical sister gate as a Python module."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "gate_mlx_candidate_contest_equivalence_z6", GATE_TOOL
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gate_mlx_candidate_contest_equivalence_z6"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_gate_tool_exists():
    assert GATE_TOOL.is_file(), f"sister gate not found at {GATE_TOOL}"


def test_canonical_constants():
    mod = _load_module()
    assert mod.Z6PCWM1_MAGIC == b"Z6WM"
    assert mod.DEFAULT_GATE_THRESHOLD == 0.001
    assert mod.EMPIRICAL_ANCHOR_DRIFT_PR95 == 0.000011
    assert mod.SCHEMA_VERSION == "mlx_candidate_contest_equivalence_gate_z6pcwm1_v1"


def test_canonical_public_api_via_dunder_all():
    mod = _load_module()
    expected = {
        "DEFAULT_GATE_THRESHOLD",
        "EMPIRICAL_ANCHOR_DRIFT_PR95",
        "SCHEMA_VERSION",
        "Z6PCWM1_MAGIC",
        "main",
        "measure_z6_decoder_parity",
    }
    assert set(mod.__all__) == expected


def test_read_archive_bytes_raw_z6pcwm1(tmp_path):
    mod = _load_module()
    if not CANONICAL_Z6_L1_ARCHIVE.is_file():
        pytest.skip(f"canonical Z6 L1 archive not present at {CANONICAL_Z6_L1_ARCHIVE}")
    raw, source = mod._read_archive_bytes(CANONICAL_Z6_L1_ARCHIVE)
    assert raw[:4] == b"Z6WM"
    assert source == "raw_z6pcwm1_bytes"


def test_read_archive_bytes_zipped(tmp_path):
    mod = _load_module()
    if not CANONICAL_Z6_L1_ARCHIVE.is_file():
        pytest.skip(f"canonical Z6 L1 archive not present at {CANONICAL_Z6_L1_ARCHIVE}")
    zip_path = tmp_path / "archive.zip"
    info = zipfile.ZipInfo("0.bin", date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(info, CANONICAL_Z6_L1_ARCHIVE.read_bytes())
    raw, source = mod._read_archive_bytes(zip_path)
    assert raw[:4] == b"Z6WM"
    assert source.startswith("zip_member_0_bin_size_")


def test_read_archive_bytes_rejects_neither_raw_nor_zip(tmp_path):
    mod = _load_module()
    bad = tmp_path / "garbage.bin"
    bad.write_bytes(b"ABCD" + b"\x00" * 100)
    with pytest.raises(ValueError, match="neither raw Z6PCWM1"):
        mod._read_archive_bytes(bad)


def test_read_archive_bytes_rejects_zip_without_0bin(tmp_path):
    mod = _load_module()
    bad_zip = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad_zip, "w") as zf:
        zf.writestr("other_member.bin", b"\x00" * 100)
    with pytest.raises(ValueError, match="missing required member '0.bin'"):
        mod._read_archive_bytes(bad_zip)


def test_read_archive_bytes_rejects_zip_with_wrong_magic(tmp_path):
    mod = _load_module()
    bad_zip = tmp_path / "bad_magic.zip"
    with zipfile.ZipFile(bad_zip, "w") as zf:
        zf.writestr("0.bin", b"WRONG_MAGIC" + b"\x00" * 100)
    with pytest.raises(ValueError, match="does not match"):
        mod._read_archive_bytes(bad_zip)


def test_measure_z6_decoder_parity_on_canonical_l1_archive():
    """Live-repo regression guard: D=Z6 L1 archive must produce PASS verdict."""
    if not CANONICAL_Z6_L1_ARCHIVE.is_file():
        pytest.skip(f"canonical Z6 L1 archive not present at {CANONICAL_Z6_L1_ARCHIVE}")
    mod = _load_module()
    result = mod.measure_z6_decoder_parity(CANONICAL_Z6_L1_ARCHIVE, n_pairs=10)
    assert result["max_abs_drift"] < mod.DEFAULT_GATE_THRESHOLD, (
        f"D=Z6 L1 archive must PASS canonical gate; got "
        f"max_abs_drift={result['max_abs_drift']} >= {mod.DEFAULT_GATE_THRESHOLD}"
    )
    # Sanity: the L1 archive has 50 pairs
    assert result["n_pairs_available"] == 50
    assert result["n_pairs_measured"] == 10
    # Sanity: decoder output [0,1] sigmoid space
    assert result["decoder_output_space"] == "sigmoid_0_to_1"
    # Sanity: frame shape is (N, 2, 3, H, W) at archive's output_hw=(48,64)
    assert result["frame_shape"] == [10, 2, 3, 48, 64]


def test_cli_pass_on_canonical_l1_archive(tmp_path):
    """CLI exit code 0 PASS on D=Z6 L1 archive."""
    if not CANONICAL_Z6_L1_ARCHIVE.is_file():
        pytest.skip(f"canonical Z6 L1 archive not present at {CANONICAL_Z6_L1_ARCHIVE}")
    output_json = tmp_path / "verdict.json"
    proc = subprocess.run(
        [
            PY,
            str(GATE_TOOL),
            "--archive",
            str(CANONICAL_Z6_L1_ARCHIVE),
            "--n-pairs",
            "5",
            "--output-json",
            str(output_json),
            "--candidate-label",
            "test_canonical_l1",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        f"expected PASS exit 0; got {proc.returncode}\nstdout:\n{proc.stdout}\n"
        f"stderr:\n{proc.stderr}"
    )
    assert output_json.is_file()
    verdict = json.loads(output_json.read_text())
    assert verdict["verdict"] == "PASS"
    assert verdict["candidate_label"] == "test_canonical_l1"
    assert verdict["max_abs_drift_decoder_parity"] < 0.001


def test_cli_pass_on_zipped_archive(tmp_path):
    """CLI exit code 0 PASS on zipped Z6PCWM1 archive."""
    if not CANONICAL_Z6_L1_ARCHIVE.is_file():
        pytest.skip(f"canonical Z6 L1 archive not present at {CANONICAL_Z6_L1_ARCHIVE}")
    zip_path = tmp_path / "z6_archive.zip"
    info = zipfile.ZipInfo("0.bin", date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(info, CANONICAL_Z6_L1_ARCHIVE.read_bytes())
    output_json = tmp_path / "verdict_zipped.json"
    proc = subprocess.run(
        [
            PY,
            str(GATE_TOOL),
            "--archive",
            str(zip_path),
            "--n-pairs",
            "5",
            "--output-json",
            str(output_json),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        f"expected PASS exit 0 on zipped archive; got {proc.returncode}\n"
        f"stderr:\n{proc.stderr}"
    )
    verdict = json.loads(output_json.read_text())
    assert verdict["verdict"] == "PASS"
    assert verdict["measurement"]["archive_source"].startswith("zip_member_0_bin_size_")


def test_cli_fail_on_missing_archive(tmp_path):
    """CLI exit code 2 when --archive does not exist."""
    output_json = tmp_path / "verdict.json"
    proc = subprocess.run(
        [
            PY,
            str(GATE_TOOL),
            "--archive",
            str(tmp_path / "does_not_exist.bin"),
            "--output-json",
            str(output_json),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2
    assert "does not exist" in proc.stderr


def test_cli_fail_on_bad_threshold(tmp_path):
    """CLI exit code 2 when threshold <= 0."""
    output_json = tmp_path / "verdict.json"
    fake_archive = tmp_path / "fake.bin"
    fake_archive.write_bytes(b"Z6WM" + b"\x00" * 100)
    proc = subprocess.run(
        [
            PY,
            str(GATE_TOOL),
            "--archive",
            str(fake_archive),
            "--output-json",
            str(output_json),
            "--gate-threshold-decoder-parity",
            "0",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2


def test_canonical_provenance_markers_in_verdict(tmp_path):
    """Per Catalog #287/#323/#192/#1/#317/#341 non-promotable markers."""
    if not CANONICAL_Z6_L1_ARCHIVE.is_file():
        pytest.skip(f"canonical Z6 L1 archive not present at {CANONICAL_Z6_L1_ARCHIVE}")
    output_json = tmp_path / "verdict_provenance.json"
    proc = subprocess.run(
        [
            PY,
            str(GATE_TOOL),
            "--archive",
            str(CANONICAL_Z6_L1_ARCHIVE),
            "--n-pairs",
            "5",
            "--output-json",
            str(output_json),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    verdict = json.loads(output_json.read_text())
    # Canonical non-promotable markers per Catalog #287/#323/#192/#1/#317/#341
    assert verdict["axis_tag"] == "[macOS-MLX research-signal]"
    assert verdict["evidence_grade"] == "macOS-MLX-research-signal"
    assert verdict["score_claim"] is False
    assert verdict["promotion_eligible"] is False
    assert verdict["rank_or_kill_eligible"] is False
    assert verdict["ready_for_exact_eval_dispatch"] is False
    assert verdict["promotable"] is False
    # Canonical blockers explicit
    blockers = verdict["blockers"]
    assert "macos_mlx_research_signal_not_contest_authority" in blockers
    assert any(
        "requires_paired_contest_cpu_plus_cuda" in b for b in blockers
    )
    # Canonical Provenance per Catalog #323
    prov = verdict["provenance"]
    assert prov is not None
    assert prov["artifact_kind"] == "predicted_from_model"
    assert prov["measurement_axis"] == "[macOS-MLX research-signal]"
    assert prov["hardware_substrate"] == "darwin_arm64_apple_silicon"
    assert prov["promotion_eligible"] is False
    assert prov["score_claim_valid"] is False
    # Canonical grammar identification
    assert verdict["candidate_grammar"] == "Z6PCWM1"
    assert verdict["candidate_substrate_id"] == "time_traveler_l5_z6"
    assert verdict["candidate_substrate_class"] == "predictive_coding_world_model"


def test_canonical_anchor_block_cites_canonical_landings(tmp_path):
    """Verdict.canonical_anchor block cites all required canonical landings."""
    if not CANONICAL_Z6_L1_ARCHIVE.is_file():
        pytest.skip(f"canonical Z6 L1 archive not present at {CANONICAL_Z6_L1_ARCHIVE}")
    output_json = tmp_path / "verdict_anchors.json"
    proc = subprocess.run(
        [
            PY,
            str(GATE_TOOL),
            "--archive",
            str(CANONICAL_Z6_L1_ARCHIVE),
            "--n-pairs",
            "5",
            "--output-json",
            str(output_json),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    verdict = json.loads(output_json.read_text())
    anchor = verdict["canonical_anchor"]
    assert anchor["pr95_empirical_anchor_drift"] == 0.000011
    assert anchor["z6_l1_promotion_commit"] == "8833b9db5"
    assert anchor["pr95_canonical_gate_commit"] == "69c316ca4"
    assert anchor["cascade_doctrine_commit"] == "fb270e9b6"
    assert anchor["mlx_first_doctrine_commit"] == "4107bbf8d"
    assert (
        anchor["sister_lane_id"]
        == "lane_path_3_sister_1265_gate_z6pcwm1_grammar_20260526"
    )
    # Scope note must reference Yousfi dissent + Catalog #164 + #226
    assert "Steps 1-2" in verdict["scope_note"]
    assert "Catalog #164" in verdict["scope_note"] or "164" in verdict["scope_note"]


def test_synthetic_drift_perturbation_flips_pass_to_fail(tmp_path):
    """Perturbing the decoder state_dict should push max_abs_drift above 0.001.

    This validates that the gate is actually sensitive to MLX↔PyTorch drift
    rather than vacuously passing on any input. We construct a perturbed
    archive by modifying ONLY the MLX-side decoder weights post-archive-load
    and re-rendering; the gate's verdict should FAIL.
    """
    if not CANONICAL_Z6_L1_ARCHIVE.is_file():
        pytest.skip(f"canonical Z6 L1 archive not present at {CANONICAL_Z6_L1_ARCHIVE}")
    mod = _load_module()
    archive_bytes = CANONICAL_Z6_L1_ARCHIVE.read_bytes()
    pytorch_model, _arc = mod._build_pytorch_substrate_from_archive(archive_bytes)
    mlx_renderer = mod._build_mlx_renderer_from_archive(archive_bytes)

    # Perturb the MLX decoder's final-conv weight bias to introduce visible
    # drift. The decoder uses sigmoid output so a bias shift of +0.5 maps to
    # ~0.12 drift in [0,1] space — well above the 0.001 gate threshold.
    import mlx.core as mx
    import numpy as np

    final_idx = mlx_renderer.decoder._final_conv_index
    final_conv = getattr(mlx_renderer.decoder, f"_block_conv_{final_idx}")
    perturbed_bias = np.asarray(final_conv.bias) + 0.5
    final_conv.bias = mx.array(perturbed_bias)

    pair_indices = list(range(5))
    pytorch_frames = mod._render_pair_batch_pytorch(pytorch_model, pair_indices)
    mlx_frames = mod._render_pair_batch_mlx(mlx_renderer, pair_indices)
    drift = float(np.abs(pytorch_frames - mlx_frames).max())
    # The perturbation must produce drift above the gate threshold
    assert drift >= 0.001, (
        f"perturbed MLX decoder should produce drift >= 0.001; got {drift}"
    )


def test_cli_n_pairs_must_be_positive(tmp_path):
    """CLI exit code 2 when --n-pairs <= 0."""
    if not CANONICAL_Z6_L1_ARCHIVE.is_file():
        pytest.skip(f"canonical Z6 L1 archive not present at {CANONICAL_Z6_L1_ARCHIVE}")
    output_json = tmp_path / "verdict.json"
    proc = subprocess.run(
        [
            PY,
            str(GATE_TOOL),
            "--archive",
            str(CANONICAL_Z6_L1_ARCHIVE),
            "--output-json",
            str(output_json),
            "--n-pairs",
            "0",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2


def test_measure_returns_canonical_schema_keys():
    """Verify the measurement dict has the canonical expected keys."""
    if not CANONICAL_Z6_L1_ARCHIVE.is_file():
        pytest.skip(f"canonical Z6 L1 archive not present at {CANONICAL_Z6_L1_ARCHIVE}")
    mod = _load_module()
    result = mod.measure_z6_decoder_parity(CANONICAL_Z6_L1_ARCHIVE, n_pairs=3)
    expected_keys = {
        "archive_path",
        "archive_bytes_sha256",
        "archive_bytes_size",
        "archive_source",
        "n_pairs_requested",
        "n_pairs_available",
        "n_pairs_measured",
        "frame_shape",
        "max_abs_drift",
        "mean_abs_drift",
        "per_pair_max_drift_min",
        "per_pair_max_drift_max",
        "per_pair_max_drift_mean",
        "decoder_output_space",
        "pytorch_build_seconds",
        "mlx_build_seconds",
        "pytorch_render_seconds",
        "mlx_render_seconds",
    }
    assert expected_keys.issubset(set(result.keys()))


def test_n_pairs_capped_to_archive_num_pairs():
    """Asking for more pairs than the archive contains is bounded silently."""
    if not CANONICAL_Z6_L1_ARCHIVE.is_file():
        pytest.skip(f"canonical Z6 L1 archive not present at {CANONICAL_Z6_L1_ARCHIVE}")
    mod = _load_module()
    # L1 archive has 50 pairs; request 1000 → should cap at 50
    result = mod.measure_z6_decoder_parity(CANONICAL_Z6_L1_ARCHIVE, n_pairs=1000)
    assert result["n_pairs_requested"] == 1000
    assert result["n_pairs_available"] == 50
    assert result["n_pairs_measured"] == 50
