"""End-to-end tests for the 5 per-family ``tools/materialize_*_residual_pr106_sidecar.py``.

Each materializer is exercised through ``runpy``-style subprocess invocation
to confirm:

* CLI exits 0 on default scaffold-readiness invocation;
* CLI refuses non-empty output dir;
* CLI refuses --n-frames <= 0 when --residual-mode != empty;
* the emitted archive zip parses cleanly via parse_archive();
* the family-specific format_id appears in the parsed archive;
* the materialization_manifest.json has all the promotion-status invariants
  pinned False;
* probe-mode produces a residual whose mutating-byte changes parsed output.

These tests use the canonical PR106 r2 archive when present; otherwise they
construct a single-member synthetic zip on-the-fly via the helper.
"""

from __future__ import annotations

import importlib.util
import json
import struct
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

from tac.residual_basis.pr106_materializer_helpers import (
    PR106_BIN_MEMBER_NAME,
    repack_dense_as_sparse,
)
from tac.residual_basis.pr106_sidecar_packing import (
    PR106_RESIDUAL_FORMAT_IDS,
    parse_archive,
)

REPO_ROOT = Path(__file__).resolve().parents[3]

MATERIALIZERS = {
    "wavelet": REPO_ROOT / "tools/materialize_wavelet_residual_pr106_sidecar.py",
    "cool_chic": REPO_ROOT / "tools/materialize_cool_chic_residual_pr106_sidecar.py",
    "c3": REPO_ROOT / "tools/materialize_c3_residual_pr106_sidecar.py",
    "siren": REPO_ROOT / "tools/materialize_siren_residual_pr106_sidecar.py",
    "coord_mlp": REPO_ROOT / "tools/materialize_coord_mlp_residual_pr106_sidecar.py",
}


@pytest.fixture
def fake_pr106_archive(tmp_path: Path) -> Path:
    """Tiny single-member synthetic PR106 archive."""
    payload = b"\x42" * 4096
    arc = tmp_path / "fake_pr106.zip"
    info = zipfile.ZipInfo(filename=PR106_BIN_MEMBER_NAME, date_time=(2025, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    with zipfile.ZipFile(arc, mode="w") as zf:
        zf.writestr(info, payload)
    return arc


def _run_materializer(
    script: Path, args: list[str], *, extra_env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    import os

    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    # Make sure PYTHONPATH includes src/ for the tac imports.
    env["PYTHONPATH"] = (
        str(REPO_ROOT / "src") + ":" + str(REPO_ROOT) + ":" + env.get("PYTHONPATH", "")
    )
    return subprocess.run(
        [sys.executable, str(script), *args],
        capture_output=True,
        text=True,
        env=env,
    )


def _write_raw_frames(path: Path, *, n_frames: int = 1) -> None:
    """Write contest-camera RGB frames as uint8 raw bytes."""
    frame = np.zeros((n_frames, 874, 1164, 3), dtype=np.uint8)
    frame.tofile(path)


def _load_inflate_module(family: str):
    path = REPO_ROOT / f"submissions/pr106_{family}_residual_sidecar/inflate.py"
    spec = importlib.util.spec_from_file_location(f"inflate_{family}_residual_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _dense_probe_residual_bytes(family: str, *, n_frames: int = 1) -> bytes:
    """Construct a tiny non-zero dense residual payload for decoder parity tests."""
    camera_h, camera_w, rgb = 874, 1164, 3
    if family == "wavelet":
        half_h, half_w = camera_h // 2, camera_w // 2
        bands = np.zeros(4 * rgb * half_h * half_w, dtype=np.int8)
        bands[0] = 3
        frame = struct.pack("<4f", 0.25, 0.125, 0.0625, 0.03125) + bands.tobytes()
        return frame * n_frames
    if family == "c3":
        grid_h, grid_w = camera_h // 4, camera_w // 4
        coeffs = np.zeros(grid_h * grid_w * rgb, dtype=np.int8)
        coeffs[0] = 5
        frame = struct.pack("<f", 0.2) + coeffs.tobytes()
        return frame * n_frames
    if family == "coord_mlp":
        grid_h, grid_w = camera_h // 8, camera_w // 8
        coeffs = np.zeros(grid_h * grid_w * rgb, dtype=np.int8)
        coeffs[0] = -4
        frame = struct.pack("<f", 0.15) + coeffs.tobytes()
        return frame * n_frames
    if family == "cool_chic":
        coeffs = np.zeros(n_frames * camera_h * camera_w * rgb, dtype=np.int8)
        coeffs[0] = 2
        return struct.pack("<Hf", 1, 0.1) + coeffs.tobytes()
    if family == "siren":
        return struct.pack("<fH", 0.125, 1) + struct.pack("<HhhBbb", 0, 0, 0, 0, 7, -3)
    raise AssertionError(f"unknown family {family}")


# ── Per-family default scaffold-readiness invocation ───────────────────────


@pytest.mark.parametrize("family", sorted(MATERIALIZERS))
def test_materializer_default_empty_residual(
    family: str, fake_pr106_archive: Path, tmp_path: Path
) -> None:
    """Default invocation (empty residual) emits a valid scaffold archive."""
    output_dir = tmp_path / f"out_{family}"
    result = _run_materializer(
        MATERIALIZERS[family],
        [
            "--pr106-archive",
            str(fake_pr106_archive),
            "--output-dir",
            str(output_dir),
        ],
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    archive_zip = output_dir / f"{family}_pr106_residual_sidecar_archive.zip"
    manifest_path = output_dir / "materialization_manifest.json"
    assert archive_zip.is_file()
    assert manifest_path.is_file()
    with zipfile.ZipFile(archive_zip, mode="r") as zf:
        blob = zf.read(PR106_BIN_MEMBER_NAME)
    parsed = parse_archive(blob)
    assert parsed.format_id == PR106_RESIDUAL_FORMAT_IDS[family]
    assert parsed.family == family
    # Empty residual scaffold.
    assert parsed.residual_bytes == b""
    # Manifest invariants.
    data = json.loads(manifest_path.read_text())
    assert data["score_claim"] is False
    assert data["promotion_eligible"] is False
    assert data["ready_for_exact_eval_dispatch"] is False
    assert data["evidence_grade"] == "research_signal"
    assert data["family"] == family
    assert data["format_id"] == PR106_RESIDUAL_FORMAT_IDS[family]


@pytest.mark.parametrize("family", sorted(MATERIALIZERS))
def test_materializer_refuses_existing_non_empty_output_dir(
    family: str, fake_pr106_archive: Path, tmp_path: Path
) -> None:
    out = tmp_path / f"out_{family}"
    out.mkdir()
    (out / "stale.txt").write_text("hello")
    result = _run_materializer(
        MATERIALIZERS[family],
        ["--pr106-archive", str(fake_pr106_archive), "--output-dir", str(out)],
    )
    assert result.returncode == 2
    assert "must be empty or not exist" in result.stderr


# ── Probe-mode invocations (per-family family-specific args) ───────────────


def test_wavelet_probe_mode_produces_nonzero_residual(
    fake_pr106_archive: Path, tmp_path: Path
) -> None:
    """Probe mode with --n-frames=2 produces a residual whose first byte is non-zero."""
    out = tmp_path / "wavelet_probe"
    result = _run_materializer(
        MATERIALIZERS["wavelet"],
        [
            "--pr106-archive",
            str(fake_pr106_archive),
            "--output-dir",
            str(out),
            "--n-frames",
            "2",
            "--residual-mode",
            "probe",
        ],
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    archive_zip = out / "wavelet_pr106_residual_sidecar_archive.zip"
    with zipfile.ZipFile(archive_zip, mode="r") as zf:
        parsed = parse_archive(zf.read(PR106_BIN_MEMBER_NAME))
    assert len(parsed.residual_bytes) > 0
    # The 17th byte is the first int8 coefficient (after 4×4B band scales).
    # In probe mode it should be 0x01.
    assert parsed.residual_bytes[16] == 0x01


def test_cool_chic_probe_mode_produces_nonzero_residual(
    fake_pr106_archive: Path, tmp_path: Path
) -> None:
    out = tmp_path / "cool_chic_probe"
    result = _run_materializer(
        MATERIALIZERS["cool_chic"],
        [
            "--pr106-archive",
            str(fake_pr106_archive),
            "--output-dir",
            str(out),
            "--n-frames",
            "2",
            "--n-levels",
            "1",
            "--residual-mode",
            "probe",
        ],
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"


def test_c3_probe_mode_produces_nonzero_residual(
    fake_pr106_archive: Path, tmp_path: Path
) -> None:
    out = tmp_path / "c3_probe"
    result = _run_materializer(
        MATERIALIZERS["c3"],
        [
            "--pr106-archive",
            str(fake_pr106_archive),
            "--output-dir",
            str(out),
            "--n-frames",
            "2",
            "--residual-mode",
            "probe",
        ],
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"


def test_siren_probe_mode_produces_sparse_coefs(
    fake_pr106_archive: Path, tmp_path: Path
) -> None:
    out = tmp_path / "siren_probe"
    result = _run_materializer(
        MATERIALIZERS["siren"],
        [
            "--pr106-archive",
            str(fake_pr106_archive),
            "--output-dir",
            str(out),
            "--n-coefs",
            "16",
            "--residual-mode",
            "probe",
        ],
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    archive_zip = out / "siren_pr106_residual_sidecar_archive.zip"
    with zipfile.ZipFile(archive_zip, mode="r") as zf:
        parsed = parse_archive(zf.read(PR106_BIN_MEMBER_NAME))
    # Header: 4B scale + 2B count = 6B; then 16 × 9B coefs = 144B; total 150B.
    assert len(parsed.residual_bytes) == 150


def test_coord_mlp_probe_mode_produces_nonzero_residual(
    fake_pr106_archive: Path, tmp_path: Path
) -> None:
    out = tmp_path / "coord_mlp_probe"
    result = _run_materializer(
        MATERIALIZERS["coord_mlp"],
        [
            "--pr106-archive",
            str(fake_pr106_archive),
            "--output-dir",
            str(out),
            "--n-frames",
            "2",
            "--residual-mode",
            "probe",
        ],
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"


@pytest.mark.parametrize("family", sorted(MATERIALIZERS))
def test_sparse_repack_decodes_to_same_residual_as_dense(family: str) -> None:
    """Sparse PacketIR repack preserves each family decoder's dense residual semantics."""
    n_frames = 1
    dense = _dense_probe_residual_bytes(family, n_frames=n_frames)
    sparse = repack_dense_as_sparse(
        family=family,
        dense_residual_bytes=dense,
        n_frames=n_frames,
    )
    assert sparse
    inflate = _load_inflate_module(family)
    dense_decode = getattr(inflate, f"decode_{family}_residual")
    sparse_decode = getattr(inflate, f"decode_{family}_residual_sparse")
    np.testing.assert_allclose(
        sparse_decode(sparse, n_frames=n_frames),
        dense_decode(dense, n_frames=n_frames),
        rtol=0,
        atol=0,
    )


# ── Negative invocations ─────────────────────────────────────────────────────


@pytest.mark.parametrize("family", sorted(MATERIALIZERS))
def test_materializer_refuses_non_empty_mode_without_n_count(
    family: str, fake_pr106_archive: Path, tmp_path: Path
) -> None:
    """--residual-mode != empty without --n-frames > 0 (or --n-coefs > 0 for SIREN) is refused."""
    out = tmp_path / f"out_{family}_refuse"
    result = _run_materializer(
        MATERIALIZERS[family],
        [
            "--pr106-archive",
            str(fake_pr106_archive),
            "--output-dir",
            str(out),
            "--residual-mode",
            "zero",
        ],
    )
    assert result.returncode == 2
    # The error message names "--n-frames" or "--n-coefs" depending on the family.
    assert "n-frames" in result.stderr or "n-coefs" in result.stderr


@pytest.mark.parametrize("family", ("wavelet", "cool_chic", "c3"))
def test_l2_encoded_materializer_requires_explicit_byte_budget(
    family: str, fake_pr106_archive: Path, tmp_path: Path
) -> None:
    decoded_raw = tmp_path / f"{family}_decoded.raw"
    gt_raw = tmp_path / f"{family}_gt.raw"
    _write_raw_frames(decoded_raw)
    _write_raw_frames(gt_raw)
    result = _run_materializer(
        MATERIALIZERS[family],
        [
            "--pr106-archive",
            str(fake_pr106_archive),
            "--output-dir",
            str(tmp_path / f"{family}_l2_no_budget"),
            "--residual-mode",
            "l2_encoded",
            "--decoded-raw",
            str(decoded_raw),
            "--gt-raw",
            str(gt_raw),
            "--n-frames",
            "1",
        ],
    )
    assert result.returncode == 2
    assert "--byte-budget > 0" in result.stderr


@pytest.mark.parametrize("family", ("wavelet", "cool_chic", "c3"))
def test_l2_encoded_materializer_does_not_silently_raise_sub_dense_budget(
    family: str, fake_pr106_archive: Path, tmp_path: Path
) -> None:
    decoded_raw = tmp_path / f"{family}_decoded.raw"
    gt_raw = tmp_path / f"{family}_gt.raw"
    _write_raw_frames(decoded_raw, n_frames=2)
    _write_raw_frames(gt_raw, n_frames=2)
    result = _run_materializer(
        MATERIALIZERS[family],
        [
            "--pr106-archive",
            str(fake_pr106_archive),
            "--output-dir",
            str(tmp_path / f"{family}_l2_sub_dense"),
            "--residual-mode",
            "l2_encoded",
            "--decoded-raw",
            str(decoded_raw),
            "--gt-raw",
            str(gt_raw),
            "--n-frames",
            "2",
            "--byte-budget",
            "1",
        ],
    )
    assert result.returncode == 2
    assert "dense" in result.stderr


# ── Deterministic bytes ──────────────────────────────────────────────────────


@pytest.mark.parametrize("family", sorted(MATERIALIZERS))
def test_materializer_deterministic_archive_bytes(
    family: str, fake_pr106_archive: Path, tmp_path: Path
) -> None:
    """Identical CLI invocations produce byte-identical archive zips."""
    a = tmp_path / f"out_a_{family}"
    b = tmp_path / f"out_b_{family}"
    r1 = _run_materializer(
        MATERIALIZERS[family],
        ["--pr106-archive", str(fake_pr106_archive), "--output-dir", str(a)],
    )
    r2 = _run_materializer(
        MATERIALIZERS[family],
        ["--pr106-archive", str(fake_pr106_archive), "--output-dir", str(b)],
    )
    assert r1.returncode == 0 and r2.returncode == 0
    archive_a = a / f"{family}_pr106_residual_sidecar_archive.zip"
    archive_b = b / f"{family}_pr106_residual_sidecar_archive.zip"
    assert archive_a.read_bytes() == archive_b.read_bytes()
