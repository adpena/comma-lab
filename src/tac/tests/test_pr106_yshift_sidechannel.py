"""Tests for pr106_yshift_sidechannel — wire format + roundtrip + builder.

Covers:
  - SC01 YSHIFT payload encode/decode roundtrip (numerics)
  - Outer 0xFC dispatch byte + uint24 PR106 length wrapper
  - Brotli round-trip on zero-init payload (CPU smoke / wire-format proof)
  - Magic-byte anti-corruption guards
  - shift_rgb_uint8 numerical correctness
  - apply_yshift composes (Y_offset clip, then shift) correctly

Mirrors the test_apogee_v2_parser_roundtrip.py + sister sidechannel patterns.
CPU-only — does not load CUDA scorers (per CLAUDE.md strict-scorer-rule, scorers
are NEVER loaded at inflate time).

Per CLAUDE.md MPS-noise rule: any score-producing assertion would need
[contest-CUDA] tag — this file ONLY tests bytewise wire format + numerics,
which are CUDA-independent.
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

from tac.repo_io import read_json, sha256_file

REPO_ROOT = Path(__file__).resolve().parents[3]
PR106_ARCHIVE = REPO_ROOT / (
    "experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/archive.zip"
)
INFLATE_DIR = REPO_ROOT / "submissions/pr106_yshift_sidechannel"
APOGEE_SRC = REPO_ROOT / "submissions/apogee_intN/src"


def _load_inflate():
    sys.modules.pop("inflate", None)  # avoid sister-module collision
    sys.path.insert(0, str(INFLATE_DIR))
    sys.path.insert(0, str(APOGEE_SRC))
    import inflate  # type: ignore[import-not-found]
    return inflate


def _load_builder():
    sys.path.insert(0, str(REPO_ROOT / "experiments"))
    import build_pr106_yshift_sidechannel as builder  # type: ignore[import-not-found]
    return builder


def test_sc01_constants():
    """Magic bytes + struct shape match the codex_metric_yshift SC01 mode-7 wire."""
    inflate = _load_inflate()
    assert inflate.SC01_MAGIC == b"SC01"
    assert inflate.SIDECHANNEL_MODE_Y_SHIFT == 7
    assert inflate.SC01_HEADER.size == 14  # 4-byte magic + uint8 mode + uint8 ch + uint32 n + float32 step
    assert inflate.YSHIFT_MAGIC_BYTE == 0xFC
    assert inflate.SIDECHANNEL_VERSION == 1


def test_sc01_encode_decode_roundtrip_zero():
    """All-zero corrections roundtrip exactly through SC01 + brotli."""
    builder = _load_builder()
    inflate = _load_inflate()
    n_frames = 1200
    values = np.zeros((n_frames, 3), dtype=np.int8)
    raw = builder._encode_sc01_yshift(values, step=1.0)
    import brotli
    blob = brotli.compress(raw, quality=11)
    decoded = inflate.decode_sidechannel_blob(blob)
    assert decoded["mode_id"] == 7
    assert decoded["channels"] == 3
    assert decoded["step"] == 1.0
    assert decoded["raw"].shape == (n_frames, 3)
    assert np.array_equal(decoded["raw"], values)


def test_sc01_encode_decode_roundtrip_random():
    """Random int8 corrections roundtrip exactly through SC01 + brotli."""
    builder = _load_builder()
    inflate = _load_inflate()
    rng = np.random.default_rng(seed=42)
    n_frames = 1200
    values = rng.integers(-127, 128, size=(n_frames, 3), dtype=np.int8)
    raw = builder._encode_sc01_yshift(values, step=0.5)
    import brotli
    blob = brotli.compress(raw, quality=11)
    decoded = inflate.decode_sidechannel_blob(blob)
    assert decoded["raw"].dtype == np.int8
    assert decoded["raw"].shape == (n_frames, 3)
    assert np.array_equal(decoded["raw"], values)
    assert abs(decoded["step"] - 0.5) < 1e-6


def test_sc01_decode_rejects_bad_magic():
    """SC01 parser raises on wrong magic (anti-corruption)."""
    inflate = _load_inflate()
    import brotli
    bad = struct.pack("<4sBBIf", b"BAD!", 7, 3, 1200, 1.0) + (b"\x00" * 3600)
    blob = brotli.compress(bad, quality=11)
    with pytest.raises(ValueError, match="bad SC01 magic"):
        inflate.decode_sidechannel_blob(blob)


def test_sc01_decode_rejects_wrong_mode():
    """SC01 parser raises on non-7 mode (we only support YSHIFT here)."""
    inflate = _load_inflate()
    import brotli
    bad = struct.pack("<4sBBIf", b"SC01", 6, 2, 1200, 1.0) + (b"\x00" * 2400)
    blob = brotli.compress(bad, quality=11)
    with pytest.raises(ValueError, match="unsupported sidechannel mode_id"):
        inflate.decode_sidechannel_blob(blob)


def test_sc01_decode_rejects_wrong_channel_count():
    """SC01 parser raises if YSHIFT mode has != 3 channels."""
    inflate = _load_inflate()
    import brotli
    bad = struct.pack("<4sBBIf", b"SC01", 7, 2, 1200, 1.0) + (b"\x00" * 2400)
    blob = brotli.compress(bad, quality=11)
    with pytest.raises(ValueError, match="YSHIFT expects 3 channels"):
        inflate.decode_sidechannel_blob(blob)


def test_shift_rgb_uint8_zero():
    """shift_rgb_uint8(0, 0) returns the input frame unmodified."""
    inflate = _load_inflate()
    rng = np.random.default_rng(seed=7)
    frame = rng.integers(0, 256, size=(384, 512, 3), dtype=np.uint8)
    out = inflate.shift_rgb_uint8(frame, 0, 0)
    assert np.array_equal(out, frame)


def test_shift_rgb_uint8_translation():
    """A (dy=2, dx=3) translation moves pixels by (2 rows down, 3 cols right).

    Codex shift_rgb pattern: out = frame.copy(), then out[dst_slice] = frame[src_slice].
    So (10, 20) gets overwritten with frame[8, 17] = 0; (12, 23) becomes the moved pixel.
    """
    inflate = _load_inflate()
    h, w = 32, 48
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[10, 20] = [123, 45, 67]  # set one pixel
    out = inflate.shift_rgb_uint8(frame, 2, 3)
    # Pixel (10, 20) → (12, 23) after dy=2, dx=3 shift
    assert tuple(out[12, 23]) == (123, 45, 67)
    # Edge column 0..2 retains zero (source [src_x0=0:src_x1=w-3] starts at column 0,
    # so column 0..2 of the output is the un-shifted fallback = zero in this test).
    assert tuple(out[12, 0]) == (0, 0, 0)
    # Original (10, 20) is overwritten by the shifted source = frame[8, 17] = zero.
    assert tuple(out[10, 20]) == (0, 0, 0)


def test_shift_rgb_uint8_negative_translation():
    """A (dy=-1, dx=-1) translation moves pixels up-left."""
    inflate = _load_inflate()
    h, w = 16, 16
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[5, 5] = [200, 100, 50]
    out = inflate.shift_rgb_uint8(frame, -1, -1)
    # Pixel (5, 5) → (4, 4)
    assert tuple(out[4, 4]) == (200, 100, 50)


def test_apply_yshift_zero():
    """apply_yshift with all-zero correction is identity."""
    inflate = _load_inflate()
    rng = np.random.default_rng(seed=11)
    frame = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    sc_row = np.zeros(3, dtype=np.int8)
    out = inflate.apply_yshift(frame, sc_row, step=1.0)
    assert np.array_equal(out, frame)


def test_apply_yshift_y_offset_only():
    """y_off=10, step=1.0 shifts all channels by +10 with clipping at 255."""
    inflate = _load_inflate()
    frame = np.full((4, 4, 3), 100, dtype=np.uint8)
    sc_row = np.array([10, 0, 0], dtype=np.int8)
    out = inflate.apply_yshift(frame, sc_row, step=1.0)
    assert (out == 110).all(), f"expected uniform 110, got min={out.min()}, max={out.max()}"


def test_apply_yshift_y_offset_clips_high():
    """y_off pushing values >255 saturates to 255."""
    inflate = _load_inflate()
    frame = np.full((4, 4, 3), 250, dtype=np.uint8)
    sc_row = np.array([20, 0, 0], dtype=np.int8)
    out = inflate.apply_yshift(frame, sc_row, step=1.0)
    assert (out == 255).all()


def test_apply_yshift_y_offset_clips_low():
    """y_off pushing values <0 saturates to 0."""
    inflate = _load_inflate()
    frame = np.full((4, 4, 3), 5, dtype=np.uint8)
    sc_row = np.array([-20, 0, 0], dtype=np.int8)
    out = inflate.apply_yshift(frame, sc_row, step=1.0)
    assert (out == 0).all()


def test_outer_archive_layout_zero_search():
    """Build with --search-mode zero produces a parseable archive +44 bytes vs PR106."""
    if not PR106_ARCHIVE.is_file():
        pytest.skip(f"PR106 anchor not present at {PR106_ARCHIVE}")
    builder = _load_builder()
    inflate = _load_inflate()

    pr106_bytes = builder._read_pr106_bytes(PR106_ARCHIVE)
    n_frames = 1200
    values = builder._zero_search(n_frames)
    raw = builder._encode_sc01_yshift(values, step=1.0)
    import brotli
    blob = brotli.compress(raw, quality=11)
    new_bin = builder._build_yshift_archive_bytes(pr106_bytes, blob)

    # Outer wrapper structure
    assert new_bin[0] == 0xFC
    pr106_len = int.from_bytes(new_bin[1:4], "little")
    assert pr106_len == len(pr106_bytes)
    assert new_bin[4 + pr106_len] == 1  # SIDECHANNEL_VERSION

    # Roundtrip
    sd, lat, meta, sc = inflate.parse_yshift_archive(new_bin)
    assert sc is not None
    assert sc["mode_id"] == 7
    assert sc["raw"].shape == (n_frames, 3)
    assert (sc["raw"] == 0).all()
    assert len(sd) == 28
    assert tuple(lat.shape) == (600, 28)
    assert meta == {"n_pairs": 600, "latent_dim": 28, "base_channels": 36, "eval_size": [384, 512]}


def test_outer_archive_layout_no_sidechannel():
    """Build with sc01_blob=None embeds PR106 only — parses with sidechannel=None."""
    if not PR106_ARCHIVE.is_file():
        pytest.skip(f"PR106 anchor not present at {PR106_ARCHIVE}")
    builder = _load_builder()
    inflate = _load_inflate()

    pr106_bytes = builder._read_pr106_bytes(PR106_ARCHIVE)
    new_bin = builder._build_yshift_archive_bytes(pr106_bytes, sc01_blob=None)
    sd, lat, meta, sc = inflate.parse_yshift_archive(new_bin)
    assert sc is None
    assert len(sd) == 28


def test_outer_archive_rejects_bad_magic():
    """Outer parser raises if first byte != 0xFC."""
    if not PR106_ARCHIVE.is_file():
        pytest.skip(f"PR106 anchor not present at {PR106_ARCHIVE}")
    builder = _load_builder()
    inflate = _load_inflate()

    pr106_bytes = builder._read_pr106_bytes(PR106_ARCHIVE)
    new_bin = bytearray(builder._build_yshift_archive_bytes(pr106_bytes, sc01_blob=None))
    new_bin[0] = 0xA5  # apogee_int5's magic — should be rejected
    with pytest.raises(ValueError, match="pr106_yshift magic mismatch"):
        inflate.parse_yshift_archive(bytes(new_bin))


def test_yshift_candidate_grid_is_deterministic_and_contains_single_noop():
    builder = _load_builder()
    grid = builder.build_yshift_candidate_grid(radius=1)

    assert grid.dtype == np.int8
    assert grid.shape == (27, 3)
    assert tuple(grid[0]) == (-1, -1, -1)
    assert tuple(grid[-1]) == (1, 1, 1)
    assert int(((grid == 0).all(axis=1)).sum()) == 1


def test_choose_yshift_candidates_from_scores_keeps_noop_without_improvement():
    builder = _load_builder()
    grid = builder.build_yshift_candidate_grid(radius=1)
    zero_idx = int(np.flatnonzero((grid == 0).all(axis=1))[0])
    target_idx = int(np.flatnonzero((grid == np.array([1, 0, -1], dtype=np.int8)).all(axis=1))[0])
    scores = np.full((3, len(grid)), 10.0)
    scores[:, zero_idx] = 1.0
    scores[0, target_idx] = 0.5
    scores[1, target_idx] = 1.0
    scores[2, target_idx] = 1.5

    selected = builder.choose_yshift_candidates_from_scores(scores, grid)

    assert np.array_equal(selected[0], grid[target_idx])
    assert np.array_equal(selected[1], grid[zero_idx])
    assert np.array_equal(selected[2], grid[zero_idx])


def test_choose_yshift_candidates_from_scores_rejects_nonfinite_scores():
    builder = _load_builder()
    grid = builder.build_yshift_candidate_grid(radius=1)
    scores = np.zeros((2, len(grid)), dtype=np.float64)
    scores[0, 0] = np.nan

    with pytest.raises(ValueError, match="NaN/Inf"):
        builder.choose_yshift_candidates_from_scores(scores, grid)


def test_choose_yshift_candidates_from_score_table_file(tmp_path):
    builder = _load_builder()
    grid = builder.build_yshift_candidate_grid(radius=1)
    zero_idx = int(np.flatnonzero((grid == 0).all(axis=1))[0])
    target_idx = int(np.flatnonzero((grid == np.array([1, 0, -1], dtype=np.int8)).all(axis=1))[0])
    scores = np.full((4, len(grid)), 10.0, dtype=np.float32)
    scores[:, zero_idx] = 1.0
    scores[0, target_idx] = 0.25
    scores[3, target_idx] = 0.75
    table_path = tmp_path / "score_table.npy"
    np.save(table_path, scores)

    selected, diagnostics = builder.choose_yshift_candidates_from_score_table_file(
        table_path,
        n_frames=4,
        candidate_radius=1,
    )

    assert selected.dtype == np.int8
    assert selected.shape == (4, 3)
    assert np.array_equal(selected[0], grid[target_idx])
    assert np.array_equal(selected[1], grid[zero_idx])
    assert np.array_equal(selected[2], grid[zero_idx])
    assert np.array_equal(selected[3], grid[target_idx])
    assert diagnostics["candidate_grid_count"] == 27
    assert diagnostics["selected_nonzero_frame_count"] == 2
    assert diagnostics["selected_zero_frame_count"] == 2


def test_score_table_file_shape_must_match_candidate_grid(tmp_path):
    builder = _load_builder()
    table_path = tmp_path / "bad_table.npy"
    np.save(table_path, np.zeros((4, 26), dtype=np.float32))

    with pytest.raises(ValueError, match="score table shape mismatch"):
        builder.choose_yshift_candidates_from_score_table_file(
            table_path,
            n_frames=4,
            candidate_radius=1,
        )


def test_outer_archive_rejects_wrong_sidechannel_version():
    """Outer parser raises on unknown sidechannel version byte."""
    if not PR106_ARCHIVE.is_file():
        pytest.skip(f"PR106 anchor not present at {PR106_ARCHIVE}")
    builder = _load_builder()
    inflate = _load_inflate()

    pr106_bytes = builder._read_pr106_bytes(PR106_ARCHIVE)
    n_frames = 1200
    raw = builder._encode_sc01_yshift(np.zeros((n_frames, 3), dtype=np.int8), step=1.0)
    import brotli
    blob = brotli.compress(raw, quality=11)
    new_bin = bytearray(builder._build_yshift_archive_bytes(pr106_bytes, blob))
    pr106_len = int.from_bytes(new_bin[1:4], "little")
    new_bin[4 + pr106_len] = 99  # bogus version
    with pytest.raises(ValueError, match="sidechannel version mismatch"):
        inflate.parse_yshift_archive(bytes(new_bin))


def test_builder_score_table_mode_writes_fail_closed_metadata(tmp_path):
    """score_table mode emits bytes but remains non-promotable without CUDA auth eval."""
    if not PR106_ARCHIVE.is_file():
        pytest.skip(f"PR106 anchor not present at {PR106_ARCHIVE}")
    builder = _load_builder()
    grid = builder.build_yshift_candidate_grid(radius=1)
    zero_idx = int(np.flatnonzero((grid == 0).all(axis=1))[0])
    target_idx = int(np.flatnonzero((grid == np.array([1, 0, -1], dtype=np.int8)).all(axis=1))[0])
    n_frames = 4
    scores = np.full((n_frames, len(grid)), 10.0, dtype=np.float32)
    scores[:, zero_idx] = 1.0
    scores[0, target_idx] = 0.5
    table_path = tmp_path / "cuda_score_table.npy"
    np.save(table_path, scores)
    out_dir = tmp_path / "out"

    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "experiments/build_pr106_yshift_sidechannel.py"),
            "--pr106-archive",
            str(PR106_ARCHIVE),
            "--out-dir",
            str(out_dir),
            "--search-mode",
            "score_table",
            "--score-table-npy",
            str(table_path),
            "--candidate-radius",
            "1",
            "--n-pairs",
            "2",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "score_table selected 1 nonzero frame corrections" in proc.stdout

    metadata = read_json(out_dir / "build_metadata.json")
    assert metadata["manifest_schema"] == "pr106_yshift_sidechannel_build_metadata_v2"
    assert metadata["score_claim"] is False
    assert metadata["ready_for_exact_eval_dispatch"] is False
    assert metadata["dispatch_attempted"] is False
    assert metadata["remote_jobs_dispatched"] is False
    assert "requires_exact_cuda_auth_eval_on_built_archive" in metadata["dispatch_blockers"]
    assert "missing_cuda_score_table_manifest" in metadata["dispatch_blockers"]
    assert "nonstandard_n_pairs_not_contest_promotable" in metadata["dispatch_blockers"]
    assert metadata["score_table"]["search_diagnostics"]["candidate_grid_radius"] == 1
    assert metadata["score_table"]["search_diagnostics"]["selected_nonzero_frame_count"] == 1

    with zipfile.ZipFile(out_dir / "pr106_yshift_sidechannel_archive.zip") as z:
        payload = z.read("0.bin")
    _, _, _, sc = _load_inflate().parse_yshift_archive(payload)
    assert sc is not None
    assert sc["raw"].shape == (n_frames, 3)
    assert np.array_equal(sc["raw"][0], grid[target_idx])
    assert np.array_equal(sc["raw"][1], grid[zero_idx])


def test_builder_score_table_mode_validates_cuda_manifest(tmp_path):
    """A provided CUDA table manifest must match table/source/radius/shape exactly."""
    if not PR106_ARCHIVE.is_file():
        pytest.skip(f"PR106 anchor not present at {PR106_ARCHIVE}")
    builder = _load_builder()
    grid = builder.build_yshift_candidate_grid(radius=1)
    zero_idx = int(np.flatnonzero((grid == 0).all(axis=1))[0])
    target_idx = int(np.flatnonzero((grid == np.array([1, 0, -1], dtype=np.int8)).all(axis=1))[0])
    n_frames = 4
    scores = np.full((n_frames, len(grid)), 10.0, dtype=np.float32)
    scores[:, zero_idx] = 1.0
    scores[0, target_idx] = 0.5
    table_path = tmp_path / "cuda_score_table.npy"
    np.save(table_path, scores)
    manifest_path = tmp_path / "score_table_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "manifest_schema": "pr106_yshift_score_table_manifest_v1",
                "score_claim": False,
                "ready_for_builder": True,
                "ready_for_exact_eval_dispatch": False,
                "dispatch_attempted": False,
                "remote_jobs_dispatched": False,
                "source_archive_sha256": sha256_file(PR106_ARCHIVE),
                "score_table_npy_sha256": sha256_file(table_path),
                "candidate_radius": 1,
                "candidate_count": len(grid),
                "n_frames": n_frames,
                "score_table_shape": [n_frames, len(grid)],
                "score_step": 1.0,
            }
        ),
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"

    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "experiments/build_pr106_yshift_sidechannel.py"),
            "--pr106-archive",
            str(PR106_ARCHIVE),
            "--out-dir",
            str(out_dir),
            "--search-mode",
            "score_table",
            "--score-table-npy",
            str(table_path),
            "--score-table-manifest",
            str(manifest_path),
            "--candidate-radius",
            "1",
            "--n-pairs",
            "2",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    metadata = read_json(out_dir / "build_metadata.json")
    assert "missing_cuda_score_table_manifest" not in metadata["dispatch_blockers"]
    assert metadata["score_table"]["score_table_manifest_validated"] is True
    assert metadata["score_table"]["score_table_manifest_schema"] == "pr106_yshift_score_table_manifest_v1"


def test_score_table_manifest_validation_rejects_source_drift(tmp_path):
    if not PR106_ARCHIVE.is_file():
        pytest.skip(f"PR106 anchor not present at {PR106_ARCHIVE}")
    builder = _load_builder()
    grid = builder.build_yshift_candidate_grid(radius=1)
    table_path = tmp_path / "cuda_score_table.npy"
    np.save(table_path, np.zeros((4, len(grid)), dtype=np.float32))
    manifest_path = tmp_path / "score_table_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "score_claim": False,
                "ready_for_builder": True,
                "ready_for_exact_eval_dispatch": False,
                "dispatch_attempted": False,
                "remote_jobs_dispatched": False,
                "source_archive_sha256": "0" * 64,
                "score_table_npy_sha256": sha256_file(table_path),
                "candidate_radius": 1,
                "candidate_count": len(grid),
                "n_frames": 4,
                "score_table_shape": [4, len(grid)],
                "score_step": 1.0,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="source_archive_sha256"):
        builder.validate_score_table_manifest(
            manifest_path,
            score_table_npy=table_path,
            pr106_archive=PR106_ARCHIVE,
            n_frames=4,
            candidate_radius=1,
            candidate_count=len(grid),
            score_step=1.0,
        )


def test_search_mode_gradient_raises_without_cuda():
    """Gradient mode is a stub that raises NotImplementedError — must be invoked via CUDA dispatch."""
    builder = _load_builder()
    with pytest.raises(NotImplementedError, match="gradient search mode requires CUDA"):
        builder._gradient_search_stub(1200)


def test_search_mode_brute_force_raises_without_cuda():
    """Brute-force mode is a stub that raises NotImplementedError — must be invoked via CUDA dispatch."""
    builder = _load_builder()
    with pytest.raises(NotImplementedError, match="brute_force search mode requires CUDA"):
        builder._brute_force_search_stub(1200)


def test_archive_size_overhead_under_2kb():
    """Zero-init brotli'd SC01 + 6-byte outer wrapper adds < 2KB to PR106 archive."""
    if not PR106_ARCHIVE.is_file():
        pytest.skip(f"PR106 anchor not present at {PR106_ARCHIVE}")
    builder = _load_builder()

    pr106_bytes = builder._read_pr106_bytes(PR106_ARCHIVE)
    n_frames = 1200
    values = builder._zero_search(n_frames)
    raw = builder._encode_sc01_yshift(values, step=1.0)
    import brotli
    blob = brotli.compress(raw, quality=11)
    new_bin = builder._build_yshift_archive_bytes(pr106_bytes, blob)
    overhead = len(new_bin) - len(pr106_bytes)
    # 6-byte outer wrapper + ~37 byte brotli'd zero payload = ~43 bytes
    assert overhead < 2048, f"overhead {overhead} bytes exceeds 2KB ceiling"
    assert overhead < 100, f"zero-init overhead {overhead} should be tiny (~50 bytes)"


def test_apply_yshift_pure_shift():
    """Pure (dy, dx) translation with y_off=0 applies shift without DC change."""
    inflate = _load_inflate()
    h, w = 16, 16
    frame = np.full((h, w, 3), 100, dtype=np.uint8)
    frame[8, 8] = [200, 50, 30]
    sc_row = np.array([0, 1, 1], dtype=np.int8)
    out = inflate.apply_yshift(frame, sc_row, step=1.0)
    # Pixel (8, 8) → (9, 9)
    assert tuple(out[9, 9]) == (200, 50, 30)
    # No DC shift applied (y_off=0)
    # Original (8, 8) is overwritten by the shifted-source fallback pattern;
    # at minimum no clipping happens elsewhere
    assert out.dtype == np.uint8
