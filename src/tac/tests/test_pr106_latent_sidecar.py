"""Test lane_pr106_latent_sidecar wire formats + parser invariants.

Lane PR106-latent-sidecar gates against:

  1. Sidecar (dim, delta_q) blob encode/decode round-trip is bit-exact.
  2. Wrapper archive (PR106 + sidecar) parses back to byte-identical PR106.
  3. dim=255 sentinel maps to a no-op application.
  4. Magic byte / format_id mismatches raise ValueError (anti-corruption guard).
  5. Sidecar applied to latents is a small additive perturbation (not catastrophic).
  6. PR106 inner-archive parse still returns 28 tensors / 228,958 params after
     wrapping + unwrapping (skipped if PR106 archive not present locally).

These are static (no-CUDA) tests — they validate the wire format and parser logic,
not scorer-driven (dim, delta) selection. Stage 3 of remote_lane_pr106_latent_sidecar.sh
provides the contest-CUDA empirical measurement.
"""
from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
PR106_ARCHIVE = REPO_ROOT / (
    "experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/archive.zip"
)
SUBMISSION_DIR = REPO_ROOT / "submissions/pr106_latent_sidecar"


def _import_build_module():
    sys.path.insert(0, str(REPO_ROOT / "experiments"))
    sys.path.insert(0, str(SUBMISSION_DIR / "src"))
    import importlib

    return importlib.import_module("build_pr106_latent_sidecar")


def _import_inflate_module():
    sys.path.insert(0, str(SUBMISSION_DIR))
    sys.path.insert(0, str(SUBMISSION_DIR / "src"))
    import importlib

    sys.modules.pop("inflate", None)
    return importlib.import_module("inflate")


# =====================================================================
# Wire format invariants (no PR106 archive required)
# =====================================================================


def test_sidecar_corrections_roundtrip_random():
    """encode → decode preserves (dim_arr, delta_q_arr) bit-exactly for random inputs."""
    build = _import_build_module()
    rng = np.random.default_rng(seed=1234)
    n_pairs = 600
    dim_arr = rng.integers(0, 28, size=n_pairs).astype(np.uint8)
    delta_q_arr = rng.integers(-127, 128, size=n_pairs).astype(np.int8)

    blob = build.encode_sidecar_corrections(dim_arr, delta_q_arr)
    rt_dim, rt_delta = build.decode_sidecar_corrections(blob)

    # Encoder maps delta=0 → dim=255 sentinel; verify that mapping.
    expected_dim = np.where(delta_q_arr == 0, 255, dim_arr).astype(np.uint8)
    assert np.array_equal(rt_dim, expected_dim)
    assert np.array_equal(rt_delta, delta_q_arr)


def test_sidecar_no_op_when_delta_zero():
    """delta_q=0 ⇒ dim=255 sentinel in encoded blob; apply is a true no-op."""
    build = _import_build_module()
    n = 10
    dim_arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.uint8)
    delta_q_arr = np.zeros(n, dtype=np.int8)

    blob = build.encode_sidecar_corrections(dim_arr, delta_q_arr)
    rt_dim, rt_delta = build.decode_sidecar_corrections(blob)
    assert np.all(rt_dim == 255)
    assert np.all(rt_delta == 0)

    latents = torch.randn(n, 28)
    snapshot = latents.clone()
    build.apply_sidecar_corrections(latents, rt_dim, rt_delta)
    assert torch.equal(latents, snapshot), "no-op corrections mutated latents"


def test_sidecar_apply_modifies_correct_dim():
    """Non-no-op corrections add delta_q * 0.01 to the named dim only."""
    build = _import_build_module()
    n = 5
    dim_arr = np.array([0, 5, 10, 15, 20], dtype=np.uint8)
    delta_q_arr = np.array([1, -2, 50, -127, 100], dtype=np.int8)

    latents = torch.zeros(n, 28)
    build.apply_sidecar_corrections(latents, dim_arr, delta_q_arr)

    expected_deltas = delta_q_arr.astype(np.float64) * 0.01
    for p in range(n):
        d = int(dim_arr[p])
        assert abs(latents[p, d].item() - expected_deltas[p]) < 1e-6
        # All other dims still zero
        for d_other in range(28):
            if d_other != d:
                assert latents[p, d_other].item() == 0.0


def test_sidecar_archive_blob_roundtrip_synthetic():
    """build → parse on synthetic PR106-shaped bytes preserves payload bit-exactly."""
    build = _import_build_module()
    fake_pr106 = b"\xff\x00\x00\x10" + b"DEADBEEF" * 1024  # 4 + 8192 = 8196 bytes
    fake_sidecar = b"BROTLISIDECARBYTES" * 5  # 90 bytes

    archive_blob = build.build_sidecar_archive_blob(fake_pr106, fake_sidecar)
    pr106_back, sidecar_back = build.parse_sidecar_archive_blob(archive_blob)

    assert pr106_back == fake_pr106, "PR106 bytes mutated through build/parse"
    assert sidecar_back == fake_sidecar, "sidecar bytes mutated through build/parse"

    # Wire format: magic + format_id + 4B pr106_len + pr106 + 2B sidecar_len + sidecar
    expected_len = 1 + 1 + 4 + len(fake_pr106) + 2 + len(fake_sidecar)
    assert len(archive_blob) == expected_len


def test_sidecar_archive_magic_byte_check():
    """Wrong magic byte raises ValueError (anti-corruption guard)."""
    build = _import_build_module()
    fake_pr106 = b"\xff" + b"\x00" * 100
    blob = build.build_sidecar_archive_blob(fake_pr106, b"")
    bad = bytearray(blob)
    bad[0] = 0xFF  # PR106's magic; should be rejected
    with pytest.raises(ValueError, match="sidecar magic mismatch"):
        build.parse_sidecar_archive_blob(bytes(bad))


def test_sidecar_archive_format_id_check():
    """Wrong format_id raises ValueError."""
    build = _import_build_module()
    fake_pr106 = b"\xff" + b"\x00" * 100
    blob = build.build_sidecar_archive_blob(fake_pr106, b"")
    bad = bytearray(blob)
    bad[1] = 0x99  # not 0x01
    with pytest.raises(ValueError, match="sidecar format_id mismatch"):
        build.parse_sidecar_archive_blob(bytes(bad))


def test_sidecar_archive_trailing_bytes_check():
    """Trailing bytes raise ValueError (catches silent layout drift)."""
    build = _import_build_module()
    fake_pr106 = b"\xff" + b"\x00" * 100
    blob = build.build_sidecar_archive_blob(fake_pr106, b"X")
    bad = blob + b"GARBAGE"
    with pytest.raises(ValueError, match="sidecar archive trailing"):
        build.parse_sidecar_archive_blob(bad)


def test_sidecar_archive_truncated_check():
    """Truncated archive raises ValueError before sidecar_len read."""
    build = _import_build_module()
    fake_pr106 = b"\xff" + b"\x00" * 100
    blob = build.build_sidecar_archive_blob(fake_pr106, b"X")
    # Cut after pr106_bytes but before sidecar_len
    truncated = blob[: 2 + 4 + len(fake_pr106)]
    with pytest.raises(ValueError, match="sidecar archive truncated"):
        build.parse_sidecar_archive_blob(truncated)


# =====================================================================
# Inflate-side invariants (matches build-side parser)
# =====================================================================


def test_inflate_parser_matches_build_parser():
    """submissions/pr106_latent_sidecar/inflate.py and experiments/build_pr106_latent_sidecar.py
    agree on the (PR106, sidecar) split (parser drift safety)."""
    build = _import_build_module()
    inflate = _import_inflate_module()

    fake_pr106 = b"\xff" + b"PAYLOAD" * 200
    fake_sidecar = b"SIDE" * 50
    archive_blob = build.build_sidecar_archive_blob(fake_pr106, fake_sidecar)

    p1, s1 = build.parse_sidecar_archive_blob(archive_blob)
    p2, s2 = inflate.parse_sidecar_archive(archive_blob)
    assert p1 == p2
    assert s1 == s2


def test_inflate_decoder_matches_build_decoder():
    """Both modules' decode_sidecar_corrections produce identical outputs."""
    build = _import_build_module()
    inflate = _import_inflate_module()

    rng = np.random.default_rng(seed=42)
    n_pairs = 100
    dim_arr = rng.integers(0, 28, size=n_pairs).astype(np.uint8)
    delta_q_arr = rng.integers(-50, 51, size=n_pairs).astype(np.int8)

    blob = build.encode_sidecar_corrections(dim_arr, delta_q_arr)

    d1, q1 = build.decode_sidecar_corrections(blob)
    d2, q2 = inflate.decode_sidecar_corrections(blob)
    assert np.array_equal(d1, d2)
    assert np.array_equal(q1, q2)


# =====================================================================
# Real PR106 archive integration (skipped if archive absent)
# =====================================================================


@pytest.mark.skipif(
    not PR106_ARCHIVE.is_file(),
    reason=f"PR106 archive not present at {PR106_ARCHIVE} — skipping integration tests",
)
def test_real_pr106_unwrap_roundtrip():
    """Wrapping real PR106 in sidecar archive then unwrapping preserves PR106 bit-exactly."""
    build = _import_build_module()
    with zipfile.ZipFile(PR106_ARCHIVE) as z:
        pr106_bytes = z.read("0.bin")

    # Empty sidecar (no corrections) — minimum stress test
    blob = build.build_sidecar_archive_blob(pr106_bytes, b"")
    pr106_back, sidecar_back = build.parse_sidecar_archive_blob(blob)
    assert pr106_back == pr106_bytes
    assert sidecar_back == b""

    # PR106 inner archive must still parse via PR106's own parser
    sys.path.insert(0, str(SUBMISSION_DIR / "src"))
    from codec import parse_packed_archive  # type: ignore[import-not-found]

    sd, lat, meta = parse_packed_archive(pr106_back)
    assert len(sd) == 28, f"expected 28 PR106 tensors, got {len(sd)}"
    total_params = sum(t.numel() for t in sd.values())
    assert total_params == 228958, f"expected 228,958 params, got {total_params}"
    assert tuple(lat.shape) == (600, 28), f"expected latents (600, 28), got {tuple(lat.shape)}"
    assert meta == {
        "n_pairs": 600,
        "latent_dim": 28,
        "base_channels": 36,
        "eval_size": [384, 512],
    }


@pytest.mark.skipif(
    not PR106_ARCHIVE.is_file(),
    reason=f"PR106 archive not present at {PR106_ARCHIVE} — skipping metadata smoke test",
)
def test_cpu_smoke_builder_metadata_is_dispatch_fail_closed(tmp_path: Path):
    """The local smoke artifact must satisfy custody checks without claiming score."""
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "experiments" / "build_pr106_latent_sidecar.py"),
            "--source-archive",
            str(PR106_ARCHIVE),
            "--output-dir",
            str(tmp_path),
            "--device",
            "cpu",
            "--smoke",
            "--top-k",
            "1",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "sidecar encode/decode round-trip OK" in proc.stdout

    metadata = json.loads((tmp_path / "build_metadata.json").read_text())
    archive = tmp_path / "sidecar_archive.zip"
    assert metadata["score_claim"] is False
    assert metadata["dispatch_attempted"] is False
    assert metadata["remote_jobs_dispatched"] is False
    assert metadata["promotion_eligible"] is False
    assert metadata["ready_for_exact_eval_dispatch"] is False
    assert metadata["dispatch_blockers"]
    assert metadata["wall_clock_seconds"] == 0.0
    assert metadata["wall_clock_seconds_note"] == "omitted_for_deterministic_smoke_manifest"
    assert Path(metadata["archive_path"]).resolve() == archive.resolve()
    assert metadata["archive_zip_bytes"] == archive.stat().st_size
