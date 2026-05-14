# SPDX-License-Identifier: MIT
"""Lane Ω-W-V3 stub-mode local smoke: Stages 1 + 3 + parser-roundtrip.

The Lane Ω-W-V3 wrapper (scripts/remote_lane_omega_w_v3_pr106.sh) has 4 stages:
  Stage 1 (CPU): extract PR106 HNeRV decoder from archive.zip
  Stage 2 (CUDA): build per-channel β-Fisher sensitivity map
  Stage 3 (CPU): repack via water-filling codec → apogee_v2_archive.zip
  Stage 4 (CUDA): contest_auth_eval

Stages 1 + 3 are CPU-only; with the all-ones sensitivity stub already
on disk at experiments/results/sensitivity_map_pr106_20260504_claude/
sensitivity_map_stub.pt, the 1+3 path is fully reproducible locally.

The byte output has been documented as 164,087 bytes (the stub-mode preview
in the wrapper script). If the codec ever drifts, this test breaks at CI
time, before any operator burns $0.30 on a Vast.ai dispatch that produces
a different byte sequence.

Skipped if the prerequisite artifacts are missing (e.g., when running in a
worktree without the PR106 intake).
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
PR106_ARCHIVE = REPO / "experiments" / "results" / "public_pr106_belt_and_suspenders_intake_20260504_codex" / "archive.zip"
SENSITIVITY_STUB = REPO / "experiments" / "results" / "sensitivity_map_pr106_20260504_claude" / "sensitivity_map_stub.pt"
EXTRACT_SCRIPT = REPO / "experiments" / "extract_pr106_decoder.py"
REPACK_SCRIPT = REPO / "experiments" / "repack_pr106_with_water_filling.py"

EXPECTED_APOGEE_V2_BYTES = 164087  # From scripts/remote_lane_omega_w_v3_pr106.sh stub-mode preview
EXPECTED_TOTAL_PARAMS = 228958      # From PR106 HNeRV decoder reconstruction
EXPECTED_N_TENSORS = 28
EXPECTED_LATENT_SHAPE = (600, 28)


@pytest.fixture(scope="module")
def workdir():
    if not PR106_ARCHIVE.is_file():
        pytest.skip(f"PR106 archive not on disk: {PR106_ARCHIVE.relative_to(REPO)}")
    if not SENSITIVITY_STUB.is_file():
        pytest.skip(f"sensitivity stub not on disk: {SENSITIVITY_STUB.relative_to(REPO)}")
    if not EXTRACT_SCRIPT.is_file() or not REPACK_SCRIPT.is_file():
        pytest.skip("Lane Ω-W-V3 producer scripts not on disk")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        # Stage 1
        subprocess.run(
            [sys.executable, str(EXTRACT_SCRIPT),
             "--archive", str(PR106_ARCHIVE),
             "--out-dir", str(tmp_path)],
            check=True, capture_output=True,
        )
        # Stage 3
        subprocess.run(
            [sys.executable, str(REPACK_SCRIPT),
             "--state-dict", str(tmp_path / "state_dict.pt"),
             "--sensitivity", str(SENSITIVITY_STUB),
             "--pr106-archive", str(PR106_ARCHIVE),
             "--target-bytes", "145000",
             "--out-dir", str(tmp_path),
             "--allow-stub-design-mode"],
            check=True, capture_output=True,
        )
        yield tmp_path


def test_stage1_outputs_exist(workdir):
    assert (workdir / "state_dict.pt").exists()
    assert (workdir / "latents.pt").exists()
    assert (workdir / "metadata.json").exists()


def test_stage3_apogee_v2_archive_exists(workdir):
    assert (workdir / "apogee_v2_archive.zip").exists()
    assert (workdir / "repack_metadata.json").exists()


def test_stage3_byte_exact_invariant(workdir):
    """Byte-for-byte reproducibility of the documented stub-mode archive size."""
    archive = workdir / "apogee_v2_archive.zip"
    actual = archive.stat().st_size
    assert actual == EXPECTED_APOGEE_V2_BYTES, (
        f"apogee_v2_archive.zip = {actual} bytes; expected {EXPECTED_APOGEE_V2_BYTES} "
        f"per the documented stub-mode invariant in scripts/remote_lane_omega_w_v3_pr106.sh. "
        f"A byte drift means the codec changed without the wrapper-doc + this test being updated."
    )


def test_parser_roundtrip_via_runtime_inflate_entry(workdir):
    """The apogee_v2 inflate.py runtime entry must decode the produced archive byte-faithfully."""
    sys.path.insert(0, str(REPO))
    try:
        from submissions.apogee_v2.inflate import parse_apogee_v2_archive
    finally:
        sys.path.pop(0)
    with zipfile.ZipFile(workdir / "apogee_v2_archive.zip") as z:
        bin_bytes = z.read("0.bin")
    sd, lat, meta = parse_apogee_v2_archive(bin_bytes)
    assert len(sd) == EXPECTED_N_TENSORS
    assert tuple(lat.shape) == EXPECTED_LATENT_SHAPE
    total_params = sum(t.numel() for t in sd.values())
    assert total_params == EXPECTED_TOTAL_PARAMS, (
        f"reconstructed {total_params:,} params; expected {EXPECTED_TOTAL_PARAMS:,}"
    )
    assert meta["n_pairs"] == 600
    assert meta["latent_dim"] == 28
    assert meta["base_channels"] == 36
    assert meta["eval_size"] == [384, 512]
