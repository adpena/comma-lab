# SPDX-License-Identifier: MIT
"""NO-FAKE tests for WAVE-1G Z8 600-pair byte-closed contest-score advisory.

Slot EEE Class 2: every test verifies ACTUAL behavior (real archive build, real
byte-closure, real contest-score arithmetic against the canonical
``upstream/evaluate.py:92`` formula) — NONE would pass if the function body were
replaced by canonical-marker emission.

The contest-score arithmetic tests exercise the REAL formula on REAL byte counts;
the end-to-end faithfulness/byte-closure test exercises the REAL archive build +
real ``export_z8hpc1_archive_bytes`` byte-closure (small pair count for speed).
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[5]
TOOL = REPO_ROOT / "tools" / "z8_600pair_byte_closed_contest_score_advisory.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("z8_contest_tool", TOOL)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Contest-score arithmetic = the canonical upstream/evaluate.py:92 formula
# ---------------------------------------------------------------------------


def test_contest_score_matches_canonical_formula_exactly():
    """score = 100*d_seg + sqrt(10*d_pose) + 25*(zip/N) — verbatim from
    upstream/evaluate.py:92. Fails if any term coefficient drifts."""
    mod = _load_tool()
    d_seg, d_pose = 0.013, 0.42
    zip_size, n = 2_155_167, 37_545_489
    out = mod.compute_contest_score(
        d_seg=d_seg, d_pose=d_pose, archive_zip_size=zip_size, n_bytes=n
    )
    expected_rate = zip_size / n
    expected = 100.0 * d_seg + math.sqrt(10.0 * d_pose) + 25.0 * expected_rate
    assert out["rate"] == pytest.approx(expected_rate, rel=1e-12)
    assert out["seg_term_100_d_seg"] == pytest.approx(100.0 * d_seg, rel=1e-12)
    assert out["pose_term_sqrt_10_d_pose"] == pytest.approx(
        math.sqrt(10.0 * d_pose), rel=1e-12
    )
    assert out["rate_term_25_rate"] == pytest.approx(25.0 * expected_rate, rel=1e-12)
    assert out["contest_score"] == pytest.approx(expected, rel=1e-12)


def test_canonical_n_bytes_matches_upstream_videos_total():
    """The canonical denominator N MUST equal the live sum of upstream/videos/*
    file sizes (== upstream/evaluate.py ``uncompressed_size``). Fails if the
    hardcoded N drifts from the real video corpus."""
    mod = _load_tool()
    videos = REPO_ROOT / "upstream" / "videos"
    live_total = sum(f.stat().st_size for f in videos.rglob("*") if f.is_file())
    assert live_total == mod.CANONICAL_N_BYTES


def test_rate_term_scales_linearly_with_archive_size_not_constant():
    """The rate term is 25 * zip/N — DOUBLING the archive size DOUBLES the rate
    term. Fails if rate were a no-op constant (the phantom-rate trap)."""
    mod = _load_tool()
    base = mod.compute_contest_score(
        d_seg=0.0, d_pose=0.0, archive_zip_size=1_000_000, n_bytes=37_545_489
    )
    doubled = mod.compute_contest_score(
        d_seg=0.0, d_pose=0.0, archive_zip_size=2_000_000, n_bytes=37_545_489
    )
    assert doubled["rate_term_25_rate"] == pytest.approx(
        2.0 * base["rate_term_25_rate"], rel=1e-9
    )
    # With d_seg=d_pose=0, the entire score IS the rate term (no phantom floor).
    assert base["contest_score"] == pytest.approx(base["rate_term_25_rate"], rel=1e-9)


def test_dominant_term_is_max_of_three_terms():
    """dominant_term must be the actual argmax of {seg, pose, rate}. Fails if it
    were a hardcoded label."""
    mod = _load_tool()
    # rate-dominated (large archive, tiny distortion)
    rate_dom = mod.compute_contest_score(
        d_seg=0.001, d_pose=0.001, archive_zip_size=150_000_000, n_bytes=37_545_489
    )
    assert rate_dom["dominant_term"] == "rate"
    # pose-dominated (tiny archive, high pose)
    pose_dom = mod.compute_contest_score(
        d_seg=0.001, d_pose=0.5, archive_zip_size=100_000, n_bytes=37_545_489
    )
    assert pose_dom["dominant_term"] == "pose"
    # seg-dominated (tiny archive, high seg)
    seg_dom = mod.compute_contest_score(
        d_seg=0.5, d_pose=0.0001, archive_zip_size=100_000, n_bytes=37_545_489
    )
    assert seg_dom["dominant_term"] == "seg"


# ---------------------------------------------------------------------------
# End-to-end: real archive build + real byte-closure (small pair count)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_end_to_end_real_archive_byte_closure_and_faithful_recon():
    """Real 4-pair archive: byte-closed archive.zip is a real on-disk file
    LARGER than the 0.bin (DEFLATE + inflate runtime + src/), recon is faithful
    (in GT range, NOT saturated 155-165), and the contest score is finite with
    the rate-term derived from the REAL zip size. Fails if any stage is faked.

    Writes under a repo-relative dir (NOT pytest tmp_path /tmp) because the
    canonical macOS-CPU advisory Provenance builder refuses transient /tmp
    source paths per Catalog #323 ``_refuse_transient_path``."""
    import shutil

    mod = _load_tool()
    video = REPO_ROOT / "upstream" / "videos" / "0.mkv"
    if not video.is_file():
        pytest.skip("upstream/videos/0.mkv not present")
    out_dir = (
        REPO_ROOT
        / "experiments"
        / "results"
        / "z8_600pair_byte_closed_contest_score_advisory_test4"
    )
    if out_dir.exists():
        shutil.rmtree(out_dir)
    try:
        res = mod.run_contest_score(
            video_path=str(video),
            num_pairs=4,
            eval_h=96,
            eval_w=128,
            out_dir=out_dir,
        )
    finally:
        if out_dir.exists():
            shutil.rmtree(out_dir)
    # Byte-closure: real archive.zip strictly larger than 0.bin (runtime + src/).
    assert res["byte_closed_archive_zip_bytes"] > res["bin_bytes"]
    assert len(res["byte_closed_archive_zip_sha256"]) == 64
    # Rate derived from the REAL zip size, not a constant.
    assert res["contest_score_breakdown"]["rate"] == pytest.approx(
        res["byte_closed_archive_zip_bytes"] / mod.CANONICAL_N_BYTES, rel=1e-9
    )
    # Render faithful: recon mean in GT range (NOT collapsed/saturated).
    fm = res["render_faithfulness"]
    assert fm["recon_mean"] < 3.0 * fm["gt_mean"]
    assert fm["recon_mean"] > fm["gt_mean"] / 3.0
    # d_seg below random chance (the wavelet codec reconstructs structure).
    assert res["distortion_net"]["mean_d_seg"] < 0.4
    # Score is finite + reconstructable from the three terms.
    bd = res["contest_score_breakdown"]
    assert math.isfinite(bd["contest_score"])
    assert bd["contest_score"] == pytest.approx(
        bd["seg_term_100_d_seg"]
        + bd["pose_term_sqrt_10_d_pose"]
        + bd["rate_term_25_rate"],
        rel=1e-9,
    )
    # Non-promotable advisory markers present.
    assert res["score_claim"] is False
    assert res["promotable"] is False
    assert res["is_contest_cpu_claim"] is False
    assert res["axis_tag"] == "[macOS-CPU advisory]"


def test_frontier_anchor_read_from_pointer_not_hardcoded():
    """The frontier anchor MUST come from the canonical pointer when present
    (pointer-only per CLAUDE.md). Fails if it always returned the literal."""
    mod = _load_tool()
    ptr = REPO_ROOT / ".omx" / "state" / "canonical_frontier_pointer.json"
    frontier = mod._frontier_cpu_anchor()
    assert frontier > 0
    if ptr.is_file():
        import json

        data = json.loads(ptr.read_text())
        cpu = data.get("our_local_frontier_contest_cpu") or {}
        pointer_score = cpu.get("score")
        if isinstance(pointer_score, (int, float)) and pointer_score > 0:
            # Must equal the live pointer value, NOT the static FRONTIER anchor.
            assert frontier == pytest.approx(float(pointer_score), rel=1e-12)
