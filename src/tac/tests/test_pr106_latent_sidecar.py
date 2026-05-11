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

import hashlib
import json
import os
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


def test_builder_prefers_portable_submission_codec_path():
    """Remote bundles must not depend on the local public-PR intake clone."""

    build = _import_build_module()
    paths = [Path(path).resolve() for path in build.PR106_SRC_CANDIDATE_PATHS]

    assert paths[0] == (SUBMISSION_DIR / "src").resolve()
    assert (paths[0] / "codec.py").is_file()
    assert (paths[0] / "model.py").is_file()


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


def test_heuristic_smoke_search_emits_nonzero_delta_for_selected_pairs():
    """The smoke search must not silently emit an all-zero sidecar."""
    build = _import_build_module()

    latents = torch.zeros(3, 28)
    latents[0, 5] = 2.0
    latents[1, 6] = -3.0
    latents[2, 7] = 0.25

    dim_arr, delta_q_arr, diagnostics = build._heuristic_self_consistency_search(
        decoder=object(),
        latents=latents,
        device=torch.device("cpu"),
        top_k=2,
    )

    assert dim_arr.tolist() == [5, 6, build.NO_OP_DIM]
    assert delta_q_arr.tolist() == [-1, 1, 0]
    assert diagnostics["n_corrections"] == 2
    assert diagnostics["n_no_op"] == 1
    assert diagnostics["nonzero_delta_count"] == 2
    assert diagnostics["delta_q_min"] == -1
    assert diagnostics["delta_q_max"] == 1


def test_latent_candidate_grid_has_single_noop_and_no_zero_deltas():
    """score_table mode must have an unambiguous no-op baseline row."""
    build = _import_build_module()
    candidates = build.build_latent_candidate_grid(latent_dim=3, delta_radius=2)

    assert candidates.dtype == np.int16
    assert candidates.shape == (1 + 3 * 4, 2)
    assert candidates[0].tolist() == [build.NO_OP_DIM, 0]
    noop = (candidates[:, 0] == build.NO_OP_DIM) & (candidates[:, 1] == 0)
    assert int(noop.sum()) == 1
    assert not ((candidates[1:, 1] == 0).any())
    assert set(candidates[1:, 0].tolist()) == {0, 1, 2}
    assert set(candidates[1:, 1].tolist()) == {-2, -1, 1, 2}


def test_score_table_reducer_requires_strict_improvement_and_top_k():
    """The score-table reducer should emit bytes only for measured improvements."""
    build = _import_build_module()
    candidates = build.build_latent_candidate_grid(latent_dim=2, delta_radius=1)
    # candidates: [noop], dim0/-1, dim0/+1, dim1/-1, dim1/+1
    scores = np.array(
        [
            [10.0, 9.5, 10.2, 9.9, 11.0],   # improve dim0/-1 by 0.5
            [10.0, 10.1, 9.2, 9.4, 10.3],   # improve dim0/+1 by 0.8
            [10.0, 10.0, 10.1, 10.2, 10.3], # tie/no strict improvement -> noop
        ],
        dtype=np.float32,
    )

    dim, delta, diagnostics = build.choose_latent_corrections_from_scores(
        scores,
        candidates,
        top_k=1,
    )

    assert dim.tolist() == [build.NO_OP_DIM, 0, build.NO_OP_DIM]
    assert delta.tolist() == [0, 1, 0]
    assert diagnostics["strict_improvement_pair_count"] == 2
    assert diagnostics["selected_nonzero_pair_count"] == 1
    assert diagnostics["selected_noop_pair_count"] == 2
    assert diagnostics["top_k_cap"] == 1


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


def test_inflate_device_auto_falls_back_to_cpu_without_cuda(monkeypatch):
    """contest-CPU auth eval must not be blocked by a CUDA-only inflate guard."""
    inflate = _import_inflate_module()
    monkeypatch.delenv("PACT_INFLATE_DEVICE", raising=False)
    monkeypatch.setattr(inflate.torch.cuda, "is_available", lambda: False)

    assert inflate.select_inflate_device() == torch.device("cpu")


def test_inflate_device_auto_prefers_cuda_when_available(monkeypatch):
    """CUDA remains the default fast path when the runtime exposes it."""
    inflate = _import_inflate_module()
    monkeypatch.delenv("PACT_INFLATE_DEVICE", raising=False)
    monkeypatch.setattr(inflate.torch.cuda, "is_available", lambda: True)

    assert inflate.select_inflate_device() == torch.device("cuda")


def test_inflate_device_rejects_mps_auth_eval(monkeypatch):
    """MPS is useful for sweeps, not auth-eval custody."""
    inflate = _import_inflate_module()
    monkeypatch.setenv("PACT_INFLATE_DEVICE", "mps")

    with pytest.raises(RuntimeError, match="mps is forbidden"):
        inflate.select_inflate_device()


def test_inflate_batch_pairs_env_validation(monkeypatch):
    """Batch-size tuning is explicit and fails closed on invalid values."""
    inflate = _import_inflate_module()

    monkeypatch.delenv("PACT_INFLATE_BATCH_PAIRS", raising=False)
    assert inflate.select_batch_pairs() == inflate.DEFAULT_BATCH_PAIRS

    monkeypatch.setenv("PACT_INFLATE_BATCH_PAIRS", "8")
    assert inflate.select_batch_pairs() == 8

    monkeypatch.setenv("PACT_INFLATE_BATCH_PAIRS", "0")
    with pytest.raises(RuntimeError, match="positive integer"):
        inflate.select_batch_pairs()


def test_inflate_sh_runs_self_contained_uploaded_runtime(tmp_path: Path):
    """Modal/custom-runtime uploads must not require repo-root submissions imports."""
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "src").mkdir()
    inflate_sh = runtime / "inflate.sh"
    inflate_sh.write_text((SUBMISSION_DIR / "inflate.sh").read_text(), encoding="utf-8")
    os.chmod(inflate_sh, 0o755)
    (runtime / "inflate.py").write_text(
        "\n".join(
            [
                "#!/usr/bin/env python",
                "from __future__ import annotations",
                "import json",
                "import sys",
                "from pathlib import Path",
                "Path(sys.argv[2]).write_bytes(b'RAW')",
                "Path(__file__).with_name('argv.json').write_text(json.dumps(sys.argv[1:]))",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    data_dir = tmp_path / "data"
    out_dir = tmp_path / "out"
    data_dir.mkdir()
    out_dir.mkdir()
    (data_dir / "0.bin").write_bytes(b"sidecar-packet")
    file_list = tmp_path / "video_names.txt"
    file_list.write_text("0.mkv\n", encoding="utf-8")

    subprocess.run(
        ["bash", str(inflate_sh), str(data_dir), str(out_dir), str(file_list)],
        cwd=tmp_path,
        env={**os.environ, "PYTHON_BIN": sys.executable},
        capture_output=True,
        text=True,
        check=True,
    )

    assert (out_dir / "0.raw").read_bytes() == b"RAW"
    assert json.loads((runtime / "argv.json").read_text()) == [
        str(data_dir / "0.bin"),
        str(out_dir / "0.raw"),
    ]


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
    assert metadata["planning_target_total_score_delta_vs_pr106"] == -0.00218
    assert metadata["predicted_total_score_delta_vs_pr106"] is None
    assert metadata["predicted_total_score"] is None
    assert metadata["wall_clock_seconds"] == 0.0
    assert metadata["wall_clock_seconds_note"] == "omitted_for_deterministic_smoke_manifest"
    assert metadata["diagnostics"]["n_corrections"] == 1
    assert metadata["diagnostics"]["nonzero_delta_count"] == 1
    assert (
        metadata["diagnostics"]["delta_q_min"] != 0
        or metadata["diagnostics"]["delta_q_max"] != 0
    )
    assert Path(metadata["archive_path"]).resolve() == archive.resolve()
    assert metadata["archive_zip_bytes"] == archive.stat().st_size
    sidecar_bin = Path(metadata["sidecar_path"])
    dim_arr, delta_q_arr = _import_build_module().decode_sidecar_corrections(
        sidecar_bin.read_bytes()
    )
    assert int(np.count_nonzero(delta_q_arr)) == 1
    assert int(np.count_nonzero(dim_arr != 255)) == 1
    with zipfile.ZipFile(archive) as zf:
        infos = zf.infolist()
    assert len(infos) == 1
    info = infos[0]
    assert info.filename == "0.bin"
    assert info.date_time == (1980, 1, 1, 0, 0, 0)
    assert info.compress_type == zipfile.ZIP_STORED
    assert (info.external_attr >> 16) == 0o644


@pytest.mark.skipif(
    not PR106_ARCHIVE.is_file(),
    reason=f"PR106 archive not present at {PR106_ARCHIVE} — skipping score-table reducer smoke test",
)
def test_score_table_builder_reduces_measured_table_without_claiming_score(tmp_path: Path):
    """score_table mode lowers measured rows into bytes but remains exact-eval gated."""
    build = _import_build_module()
    candidates = build.build_latent_candidate_grid(latent_dim=28, delta_radius=1)
    scores = np.full((600, len(candidates)), 10.0, dtype=np.float32)
    # Candidate row 1 is dim0/delta=-1, row 2 is dim0/delta=+1.
    scores[0, 1] = 9.0
    scores[1, 2] = 8.0
    scores[2, 1] = 9.5
    table_path = tmp_path / "score_table.npy"
    np.save(table_path, scores, allow_pickle=False)

    out_dir = tmp_path / "out"
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "experiments" / "build_pr106_latent_sidecar.py"),
            "--source-archive",
            str(PR106_ARCHIVE),
            "--output-dir",
            str(out_dir),
            "--device",
            "cpu",
            "--smoke",
            "--search-mode",
            "score_table",
            "--score-table-npy",
            str(table_path),
            "--top-k",
            "2",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "search complete: 2 pairs corrected" in proc.stdout

    metadata = json.loads((out_dir / "build_metadata.json").read_text())
    assert metadata["search_mode"] == "score_table"
    assert metadata["score_claim"] is False
    assert metadata["ready_for_exact_eval_dispatch"] is False
    assert metadata["score_table"]["score_table_npy_sha256"]
    assert metadata["score_table"]["score_table_manifest_validated"] is False
    assert "missing_cuda_score_table_manifest" in metadata["dispatch_blockers"]
    assert metadata["diagnostics"]["strict_improvement_pair_count"] == 3
    assert metadata["diagnostics"]["selected_nonzero_pair_count"] == 2
    assert metadata["diagnostics"]["selected_noop_pair_count"] == 598

    dim_arr, delta_q_arr = build.decode_sidecar_corrections((out_dir / "sidecar.bin").read_bytes())
    assert int(np.count_nonzero(delta_q_arr)) == 2
    assert int(np.count_nonzero(dim_arr != build.NO_OP_DIM)) == 2


@pytest.mark.skipif(
    not PR106_ARCHIVE.is_file(),
    reason=f"PR106 archive not present at {PR106_ARCHIVE} — skipping score-table manifest test",
)
def test_score_table_manifest_accepts_reframed_archive_with_same_zero_bin(tmp_path: Path):
    """Kaggle may reframe the ZIP, but the measured PR106 payload is 0.bin."""
    build = _import_build_module()
    with zipfile.ZipFile(PR106_ARCHIVE) as zf:
        zero_bin = zf.read("0.bin")

    score_table = np.zeros((600, 57), dtype=np.float32)
    table_path = tmp_path / "score_table.npy"
    np.save(table_path, score_table, allow_pickle=False)
    candidates = build.build_latent_candidate_grid(latent_dim=28, delta_radius=1)
    candidate_path = tmp_path / "candidate_grid.npy"
    np.save(candidate_path, candidates, allow_pickle=False)
    reframed_archive = tmp_path / "pr106_reframed.zip"
    with zipfile.ZipFile(reframed_archive, "w", compression=zipfile.ZIP_STORED) as zf:
        info = zipfile.ZipInfo("0.bin", date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_STORED
        info.external_attr = 0o644 << 16
        zf.writestr(info, zero_bin)

    manifest_path = tmp_path / "score_table_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "manifest_schema": "pr106_latent_score_table_manifest_v1",
                "producer": "experiments/build_pr106_latent_score_table.py",
                "score_claim": False,
                "ready_for_builder": True,
                "source_archive_sha256": hashlib.sha256(
                    reframed_archive.read_bytes()
                ).hexdigest(),
                "source_zero_bin_sha256": hashlib.sha256(zero_bin).hexdigest(),
                "score_table_npy_sha256": hashlib.sha256(table_path.read_bytes()).hexdigest(),
                "candidate_grid_sha256": build.latent_candidate_grid_npy_sha256(candidates),
                "n_pairs": 600,
                "latent_dim": 28,
                "delta_radius": 1,
                "candidate_count": 57,
                "score_table_shape": [600, 57],
                "ready_for_exact_eval_dispatch": False,
                "dispatch_attempted": False,
                "remote_jobs_dispatched": False,
            }
        )
    )

    validated = build.validate_score_table_manifest(
        manifest_path,
        score_table_npy=table_path,
        source_archive=PR106_ARCHIVE,
        n_pairs=600,
        latent_dim=28,
        delta_radius=1,
        candidate_count=57,
    )

    assert validated["validated_source_archive_sha256_match"] is False
    assert validated["validated_source_zero_bin_sha256_match"] is True


@pytest.mark.skipif(
    not PR106_ARCHIVE.is_file(),
    reason=f"PR106 archive not present at {PR106_ARCHIVE} — skipping score-table manifest test",
)
def test_score_table_manifest_rejects_different_zero_bin_when_archive_sha_mismatches(
    tmp_path: Path,
):
    """Archive SHA fallback must not accept a table measured on another payload."""
    build = _import_build_module()
    score_table = np.zeros((600, 57), dtype=np.float32)
    table_path = tmp_path / "score_table.npy"
    np.save(table_path, score_table, allow_pickle=False)
    candidates = build.build_latent_candidate_grid(latent_dim=28, delta_radius=1)

    manifest_path = tmp_path / "score_table_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "manifest_schema": "pr106_latent_score_table_manifest_v1",
                "producer": "experiments/build_pr106_latent_score_table.py",
                "score_claim": False,
                "ready_for_builder": True,
                "source_archive_sha256": "a" * 64,
                "source_zero_bin_sha256": "b" * 64,
                "score_table_npy_sha256": hashlib.sha256(table_path.read_bytes()).hexdigest(),
                "candidate_grid_sha256": build.latent_candidate_grid_npy_sha256(candidates),
                "n_pairs": 600,
                "latent_dim": 28,
                "delta_radius": 1,
                "candidate_count": 57,
                "score_table_shape": [600, 57],
                "ready_for_exact_eval_dispatch": False,
                "dispatch_attempted": False,
                "remote_jobs_dispatched": False,
            }
        )
    )

    with pytest.raises(ValueError, match="source archive payload mismatch"):
        build.validate_score_table_manifest(
            manifest_path,
            score_table_npy=table_path,
            source_archive=PR106_ARCHIVE,
            n_pairs=600,
            latent_dim=28,
            delta_radius=1,
            candidate_count=57,
        )
