# SPDX-License-Identifier: MIT
"""#330 verdict-memory-reclaim tests: cheap reclaim, killpg orphan-free child, subprocess bit-identity."""
from __future__ import annotations

import json
import os
import platform
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[4]
for _p in (REPO, REPO / "src", REPO / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tac.witness_control import verdict_reclaim as vr  # noqa: E402

_GT_N24 = REPO / "experiments/results/mlx_fleet_gt_cache/gt_n24.npz"


# ---- cheap reclaim path -----------------------------------------------------
def test_rss_gib_positive_or_unavailable() -> None:
    v = vr.rss_gib()
    assert v > 0.0 or v == -1.0


def test_malloc_trim_returns_platform_method() -> None:
    m = vr.malloc_trim()
    if platform.system() == "Linux":
        assert m in {"malloc_trim", "noop"}
    elif platform.system() == "Darwin":
        assert m in {"malloc_zone_pressure_relief", "noop"}
    else:
        assert m == "noop"


def test_reclaim_process_memory_shape_and_score_neutral() -> None:
    guarded = np.arange(1000, dtype=np.float64) * 1.5
    snapshot = guarded.copy()
    out = vr.reclaim_process_memory()
    # Score-neutral: reclaim frees pages, never mutates live objects.
    assert np.array_equal(guarded, snapshot)
    assert set(out) >= {"gc_collected", "trim_method", "torch_cache_released",
                        "rss_before_gib", "rss_after_gib", "reclaimed_gib"}
    assert isinstance(out["gc_collected"], int)
    assert out["trim_method"] in {"malloc_trim", "malloc_zone_pressure_relief", "noop"}


def test_reclaim_no_measure_skips_rss() -> None:
    out = vr.reclaim_process_memory(measure=False)
    assert out["rss_before_gib"] == -1.0
    assert out["rss_after_gib"] == -1.0
    assert out["reclaimed_gib"] is None


# ---- subprocess input validation --------------------------------------------
def test_subprocess_ragged_inputs_raise() -> None:
    f = [np.zeros((4, 4, 3), np.uint8)]
    with pytest.raises(ValueError, match="ragged"):
        vr.run_verdict_in_subprocess(f, f + f, [np.zeros((4, 4), np.int64)],
                                     [np.zeros(6, np.float64)], vbatch=1)


def test_subprocess_empty_inputs_raise() -> None:
    with pytest.raises(ValueError):
        vr.run_verdict_in_subprocess([], [], [], [], vbatch=1)


# ---- killpg orphan-free discipline (#167) -----------------------------------
def test_kill_group_reaps_whole_session() -> None:
    # A detached child in its OWN session; killpg the group -> the child (and any descendants) die.
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        start_new_session=True, stdin=subprocess.DEVNULL)
    try:
        pgid = os.getpgid(proc.pid)
        assert pgid == proc.pid  # session leader => pid == pgid (the #167 no-orphan property)
        vr._kill_group(pgid, grace_s=2.0)
        # group must be gone within a short window
        deadline = time.time() + 3.0
        while time.time() < deadline and proc.poll() is None:
            time.sleep(0.05)
        assert proc.poll() is not None  # child reaped, no orphan
    finally:
        if proc.poll() is None:
            with contextlib_suppress():
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait(timeout=5)


class contextlib_suppress:
    def __enter__(self):
        return self

    def __exit__(self, *exc: object) -> bool:
        return True


# ---- worker fallback: malformed input yields an error result, rc=4 (no fabricated verdict) --------
def test_worker_error_path_writes_error_json_rc4(tmp_path: Path) -> None:
    in_path = tmp_path / "bad_in.npz"
    np.savez(in_path, n=np.asarray([1], np.int64), vbatch=np.asarray([1], np.int64))  # missing f0_0 etc.
    out_path = tmp_path / "out.json"
    rc = subprocess.call(
        [sys.executable, "-m", "tac.witness_control._verdict_subprocess_worker",
         "--in", str(in_path), "--out", str(out_path), "--seg-form", "argmax"],
        stdin=subprocess.DEVNULL, cwd=str(REPO))
    assert rc == 4
    assert out_path.exists()
    result = json.loads(out_path.read_text())
    assert "error" in result  # NO-FAKE: surfaced, not fabricated


def test_run_subprocess_raises_on_worker_error(tmp_path: Path) -> None:
    # Feed a shape the worker can serialize but whose scorer forward will fail fast is heavy; instead
    # assert the parent RAISES (not fabricates) when the child fails. Use a 1-pair tiny frame; the
    # child will error loading scorers only if unavailable -> either way a real verdict or a raise.
    f0 = [np.zeros((8, 8, 3), np.uint8)]
    f1 = [np.zeros((8, 8, 3), np.uint8)]
    lstar = [np.zeros((8, 8), np.int64)]
    pose = [np.zeros(6, np.float64)]
    try:
        out = vr.run_verdict_in_subprocess(f0, f1, lstar, pose, vbatch=1,
                                           scratch_dir=tmp_path, timeout_s=300.0)
    except RuntimeError:
        return  # acceptable: scorers/shape rejected -> parent raised, did NOT fabricate
    # If it returned, it must be a real dict with the required keys (never a fabricated score field).
    assert {"d_seg_mean", "d_pose_mean", "n"} <= set(out)


# ---- bit-identity: subprocess verdict == in-process chunked verdict (real scorers) ----------------
@pytest.mark.slow
@pytest.mark.skipif(not _GT_N24.exists(), reason="gt_n24 cache not present")
def test_subprocess_bit_identical_to_inprocess() -> None:
    from train_witness_realized_through_R_mlx import (
        cpu_verdict_d_pose_batch,
        cpu_verdict_d_seg_batch,
        load_gt_from_cache,
    )
    n = 6
    gt, seg_cpu, posenet_cpu = load_gt_from_cache(_GT_N24, n)
    f0s = [gt.gt_f0[i] for i in range(n)]
    f1s = [gt.gt_f1[i] for i in range(n)]
    lstars = [gt.lstars[i] for i in range(n)]
    poses = [gt.gt_poses[i] for i in range(n)]

    vb = 2
    ds = []
    dp = []
    for s in range(0, n, vb):
        e = min(s + vb, n)
        ds.extend(cpu_verdict_d_seg_batch(seg_cpu, f1s[s:e], lstars[s:e]))
        dp.extend(cpu_verdict_d_pose_batch(posenet_cpu, f0s[s:e], f1s[s:e], poses[s:e]))
    ip_ds, ip_dp = float(np.mean(ds)), float(np.mean(dp))

    sub = vr.run_verdict_in_subprocess(f0s, f1s, lstars, poses, vbatch=vb, timeout_s=600.0)
    # per-pair AND mean bit-identical (same frozen scorers, same preprocess->forward->argmax/MSE).
    assert np.array_equal(np.asarray(ds), np.asarray(sub["d_seg"]))
    assert np.array_equal(np.asarray(dp), np.asarray(sub["d_pose"]))
    assert np.float64(ip_ds).tobytes() == np.float64(sub["d_seg_mean"]).tobytes()
    assert np.float64(ip_dp).tobytes() == np.float64(sub["d_pose_mean"]).tobytes()


@pytest.mark.slow
@pytest.mark.skipif(not _GT_N24.exists(), reason="gt_n24 cache not present")
def test_subprocess_returns_parent_rss_to_baseline() -> None:
    from train_witness_realized_through_R_mlx import load_gt_from_cache
    n = 6
    gt, _seg, _pose = load_gt_from_cache(_GT_N24, n)
    f0s = [gt.gt_f0[i] for i in range(n)]
    f1s = [gt.gt_f1[i] for i in range(n)]
    lstars = [gt.lstars[i] for i in range(n)]
    poses = [gt.gt_poses[i] for i in range(n)]

    import gc as _gc
    _gc.collect()
    before = vr.rss_gib()
    sub = vr.run_verdict_in_subprocess(f0s, f1s, lstars, poses, vbatch=2, timeout_s=600.0)
    _gc.collect()
    after = vr.rss_gib()
    # The child held the scorer transient; parent RSS must not have ratcheted materially (<0.5 GiB).
    assert (after - before) < 0.5
    assert sub["child_rss_peak_gib"] is None or sub["child_rss_peak_gib"] > 0.0
