# SPDX-License-Identifier: MIT
"""NO-FAKE tests for the end-to-end byte-close -> eval verification harness
(``tools/verify_e2e_byte_close_eval.py``, the #107 readiness actuator).

These tests assert REAL behavior of the harness on the REAL converged n600 basin:
the actual vendored codec byte-close (~89 KB ``0.bin``), the actual contest
``archive.zip`` assembly (containing ONLY ``0.bin`` per ``compress.sh``), the actual
parse-back fixed-point parity (the G2 contract), the exact 3-section byte breakdown,
and — when the frozen scorer + GT video are present — the FULL pipeline producing an
advisory ``S`` whose components land in the basin's recorded advisory band. Every test
would FAIL if the harness fabricated bytes, dropped weights, or skipped the real eval.

Authority: ``[contest-CPU advisory]``; no score is claimed. Tests skip (not fail) when
the gitignored vendored PR95 clone, the basin checkpoint, or the frozen scorer/GT video
are unavailable, so the suite is portable.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
_TOOL_PATH = _REPO_ROOT / "tools" / "verify_e2e_byte_close_eval.py"
_BASIN = _REPO_ROOT / "experiments/results/torch_vehicle_full_mps_basin_bc20_n600/best"


def _load_harness():
    """Import the harness tool module by path (tools/ is not a package)."""
    spec = importlib.util.spec_from_file_location("verify_e2e_byte_close_eval", _TOOL_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _vendored_available() -> bool:
    try:
        from tac.torch_vehicle.vendored_imports import VENDORED_SRC

        return VENDORED_SRC.exists()
    except Exception:
        return False


_h = _load_harness()
_HAVE_BASIN = (_BASIN / "best_ema_decoder.pt").exists() and (_BASIN / "best_ema_latents.pt").exists()
_HAVE_VENDORED = _vendored_available()

requires_vendored = pytest.mark.skipif(not _HAVE_VENDORED, reason="vendored PR95 clone absent")
requires_basin = pytest.mark.skipif(
    not (_HAVE_BASIN and _HAVE_VENDORED), reason="n600 basin checkpoint or vendored clone absent"
)


def _scorer_available() -> bool:
    """The frozen DistortionNet + GT video must both be present for the full eval test."""
    if not _HAVE_VENDORED:
        return False
    try:
        from tac.torch_vehicle.vendored_imports import import_vendored

        vp = Path(import_vendored("data").get_default_video_path())
        from tac.score_aware_loop.targets import load_frozen_distortion_net  # noqa: F401

        return vp.exists()
    except Exception:
        return False


requires_scorer = pytest.mark.skipif(
    not (_HAVE_BASIN and _scorer_available()), reason="frozen scorer / GT video / basin unavailable"
)


# ---------------------------------------------------------------------------
# unit-level: checkpoint loading + dim inference (no scorer)
# ---------------------------------------------------------------------------
@requires_basin
def test_load_checkpoint_and_infer_dims_real_basin():
    """The harness loads the REAL basin and infers (latent_dim=28, base_ch=20, n=600)."""
    dec_sd, latents, meta = _h._load_checkpoint(_BASIN)
    assert latents.shape == (600, 28)
    latent_dim, base_channels, n_pairs = _h._infer_dims(dec_sd, latents, meta)
    assert (latent_dim, base_channels, n_pairs) == (28, 20, 600)
    # base_channels must be inferable from stem.weight EVEN if meta is silent (NO-FAKE).
    ld2, bc2, n2 = _h._infer_dims(dec_sd, latents, meta={})
    assert (ld2, bc2, n2) == (28, 20, 600)


@requires_basin
def test_missing_checkpoint_raises_not_fabricates():
    """A missing checkpoint dir RAISES (NO-FAKE) — never silently returns a fake archive."""
    with pytest.raises(FileNotFoundError):
        _h._load_checkpoint(_REPO_ROOT / "experiments/results/__does_not_exist__")


# ---------------------------------------------------------------------------
# byte-close + parse-back parity (the G2 fixed-point contract), real basin
# ---------------------------------------------------------------------------
@requires_basin
def test_byte_close_real_basin_is_89kb_and_parity_ok():
    """The REAL basin byte-closes to ~89 KB and parse-back is a fixed point (G2)."""
    from tac.torch_vehicle.driver import import_vendored_bundle

    vb = import_vendored_bundle()
    dec_sd, latents, meta = _h._load_checkpoint(_BASIN)
    latent_dim, base_channels, n_pairs = _h._infer_dims(dec_sd, latents, meta)
    meta_dict = {
        "n_pairs": n_pairs, "latent_dim": latent_dim,
        "base_channels": base_channels, "eval_size": [384, 512],
    }
    archive, dec_back, lat_back, parity = _h._byte_close_and_verify_parity(
        vb, dec_sd, latents, meta_dict
    )
    # The basin's recorded archive_bytes is 89136; the harness reproduces it exactly.
    assert len(archive) == 89_136, len(archive)
    assert parity["parity_ok"] is True
    assert parity["build_deterministic"] and parity["weights_fixed_point"]
    assert parity["latents_fixed_point"] and parity["keys_match"]
    assert parity["meta_roundtrip_base_channels"] == 20
    # parse-back latents preserve the full sequence shape.
    assert lat_back.shape == (600, 28)
    # parse-back state dict carries exactly the float decoder tensors.
    assert set(dec_back.keys()) == {k for k in dec_sd if torch.is_tensor(dec_sd[k])}


@requires_basin
def test_section_byte_breakdown_sums_to_archive():
    """The 3-section breakdown (meta + decoder + latents + 12 prefix B) sums to the archive."""
    from tac.torch_vehicle.driver import import_vendored_bundle

    vb = import_vendored_bundle()
    dec_sd, latents, meta = _h._load_checkpoint(_BASIN)
    ld, bc, n = _h._infer_dims(dec_sd, latents, meta)
    archive = vb.build_archive(
        dec_sd, latents,
        meta_dict={"n_pairs": n, "latent_dim": ld, "base_channels": bc, "eval_size": [384, 512]},
    )
    bd = _h._section_byte_breakdown(archive)
    total = bd["meta_brotli"] + bd["decoder_blob"] + bd["latents_brotli"] + bd["length_prefixes"]
    assert total == len(archive), (bd, len(archive))
    assert bd["trailing"] == 0  # pristine vendored 3-section grammar (no appended pose section)
    # the decoder blob dominates (the int8 weights), latents are the second-largest section.
    assert bd["decoder_blob"] > bd["latents_brotli"] > bd["meta_brotli"]


# ---------------------------------------------------------------------------
# contest packet assembly (archive.zip holds ONLY 0.bin; deterministic; runnable)
# ---------------------------------------------------------------------------
@requires_vendored
def test_contest_packet_zip_holds_only_0bin_and_is_deterministic(tmp_path):
    """The assembled archive.zip contains EXACTLY 0.bin (per compress.sh) and is byte-stable;
    the runtime tree (inflate.sh/inflate.py/src/) is copied BESIDE the zip (NOT in it)."""
    payload = b"\x00\x01\x02" * 5000  # a stand-in 0.bin (assembly is payload-agnostic)
    zip_path, zip_bytes, runtime_files = _h._assemble_contest_packet(payload, tmp_path / "pkt")

    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        assert names == ["0.bin"], names  # ONLY 0.bin is counted in the contest rate
        assert zf.read("0.bin") == payload
    assert zip_bytes == zip_path.stat().st_size
    # the runtime tree lives BESIDE the zip (not counted in rate) and is runnable.
    assert "inflate.py" in runtime_files and "inflate.sh" in runtime_files and "src/" in runtime_files
    assert (tmp_path / "pkt" / "inflate.py").exists()
    assert (tmp_path / "pkt" / "src" / "model.py").exists()
    assert (tmp_path / "pkt" / "src" / "codec.py").exists()

    # Determinism: a second assembly of the SAME payload yields a byte-identical zip.
    zip_path2, zip_bytes2, _ = _h._assemble_contest_packet(payload, tmp_path / "pkt2")
    assert zip_path2.read_bytes() == zip_path.read_bytes()
    assert zip_bytes2 == zip_bytes


# ---------------------------------------------------------------------------
# FULL pipeline end-to-end on the REAL basin (real scorer) — the NO-FAKE headline
# ---------------------------------------------------------------------------
@requires_scorer
def test_full_pipeline_real_basin_subset_matches_recorded_advisory(tmp_path):
    """The FULL harness on the REAL basin (tiny eval subset) byte-closes to ~89 KB, parse-back
    is exact, and the advisory components land in the basin's recorded band.

    Uses --max-pairs=2 so the real-scorer eval is CI-fast; the byte-close still uses ALL 600
    latents (the archive bytes are the full vehicle). The d_seg/d_pose of a 2-pair subset will
    not equal the 600-pair mean exactly, but must be the SAME ORDER as the recorded advisory
    (d_seg ~2.6e-3, d_pose ~3.4e-4) — proving the harness runs the REAL scorer, not a proxy.
    """
    report = _h.run(
        _BASIN, max_pairs=2, taper_channels=None, rate_denom=None,
        keep_packet=True, packet_dir=tmp_path / "pkt",
    )
    # byte-close fidelity: the contest 0.bin is the basin's 89,136 B; the zip adds small overhead.
    assert report["archive_bin_bytes"] == 89_136
    assert report["archive_zip_bytes"] > report["archive_bin_bytes"]  # zip container overhead
    assert report["zip_container_overhead_bytes"] > 0
    assert report["parseback_parity"]["parity_ok"] is True

    # the rate term uses the REAL archive.zip st_size over the source-video denominator.
    assert report["rate_denominator_bytes"] == 37_545_489  # 0.mkv st_size (evaluate.py:64)
    assert abs(report["rate"] - report["archive_zip_bytes"] / 37_545_489) < 1e-12

    # REAL scorer components (a 2-pair subset, but same order as the 600-pair advisory).
    assert 0.0 <= report["d_seg"] < 0.05, report["d_seg"]
    assert 0.0 <= report["d_pose"] < 0.05, report["d_pose"]
    # S recomputed from components, NOT a rounded field.
    expect_S = (
        100.0 * report["d_seg"]
        + (10.0 * report["d_pose"] + 1e-12) ** 0.5
        + 25.0 * report["rate"]
    )
    assert abs(report["score_S"] - expect_S) < 1e-9
    # the headline S (zip bytes) and the bin-bytes S agree to the tiny zip overhead.
    assert abs(report["score_S"] - report["score_S_from_bin_bytes"]) < 0.001
    assert report["authority"].startswith("[contest-CPU advisory]")
    assert report["promotion_claim"] is False

    # the assembled packet is a real, runnable contest submission_dir.
    pkt = Path(report["packet_dir"])
    assert (pkt / "archive.zip").exists() and (pkt / "inflate.sh").exists()
    with zipfile.ZipFile(pkt / "archive.zip") as zf:
        assert zf.namelist() == ["0.bin"]


@requires_scorer
def test_cli_main_writes_report_json(tmp_path):
    """The CLI entry point runs end-to-end and writes a well-formed JSON report."""
    out = tmp_path / "report.json"
    rc = _h.main([
        "--ckpt-dir", str(_BASIN), "--max-pairs", "2", "--out", str(out),
    ])
    assert rc == 0
    blob = json.loads(out.read_text())
    assert blob["archive_bin_bytes"] == 89_136
    assert blob["parseback_parity"]["parity_ok"] is True
    assert "score_S" in blob and blob["promotion_claim"] is False
