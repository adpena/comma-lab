"""Tests for the ddm_up1 pose instrument.

The load-bearing behaviours are: seeded-random (never prefix) sampling, GT
lineage being RETURNED rather than inferred, the raw geometry guard, and the
offset parser refusing a sweep with no base control.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO / "experiments/ddm_up1_decode_axis_photometric_probe.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("ddm_up1_probe", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


up1 = _load_module()


class TestParseOffsets:
    def test_parses_a_sweep(self):
        assert up1.parse_offsets("-1,0,1") == (-1.0, 0.0, 1.0)

    def test_tolerates_whitespace_and_blanks(self):
        assert up1.parse_offsets(" -1 , 0 , , 1 ") == (-1.0, 0.0, 1.0)

    def test_refuses_a_sweep_without_the_base_control(self):
        # Without offset 0 there is no control, so no ratio is interpretable.
        with pytest.raises(up1.Up1Error, match="base control"):
            up1.parse_offsets("-1,1")

    def test_refuses_an_empty_sweep(self):
        with pytest.raises(up1.Up1Error):
            up1.parse_offsets("")


class TestSelectPairs:
    def test_is_never_a_prefix(self):
        # m88/m96: a prefix of a skewed population is a different population, and
        # on the pose axis prefixes measure 2.5-4.2x anti-conservatively.
        pairs = up1.select_pairs(64, seed=20260819)
        assert not np.array_equal(pairs, np.arange(64))
        assert pairs.max() > 400

    def test_is_seeded_and_reproducible(self):
        assert np.array_equal(up1.select_pairs(32, 7), up1.select_pairs(32, 7))

    def test_distinct_seeds_give_distinct_samples(self):
        assert not np.array_equal(up1.select_pairs(32, 7), up1.select_pairs(32, 8))

    def test_is_sorted_and_unique(self):
        pairs = up1.select_pairs(128, seed=3)
        assert np.array_equal(pairs, np.sort(pairs))
        assert len(set(pairs.tolist())) == 128

    @pytest.mark.parametrize("count", [0, -1, up1.N_PAIRS_TOTAL + 1])
    def test_refuses_out_of_range(self, count):
        with pytest.raises(up1.Up1Error):
            up1.select_pairs(count, seed=1)


class TestOpenRaw:
    def test_refuses_a_missing_file(self, tmp_path):
        with pytest.raises(up1.Up1Error, match="does not exist"):
            up1.open_raw(tmp_path / "absent.raw", verify_sha=False)

    def test_refuses_wrong_geometry(self, tmp_path):
        path = tmp_path / "short.raw"
        path.write_bytes(b"\x00" * 1024)
        with pytest.raises(up1.Up1Error, match="expected"):
            up1.open_raw(path, verify_sha=False)

    def test_refuses_an_unknown_decode_when_verifying(self, tmp_path):
        # Right geometry, wrong bytes: fail closed rather than measure a stranger.
        path = tmp_path / "wrong.raw"
        with path.open("wb") as stream:
            stream.truncate(2 * up1.N_PAIRS_TOTAL * up1.FRAME_BYTES)
        with pytest.raises(up1.Up1Error, match="neither known decode"):
            up1.open_raw(path, verify_sha=True)


class TestGtLineage:
    """The GT lineage is 99.9960% of the advisory-vs-contest pose offset (pi2)."""

    def test_npz_reports_the_av_lineage(self, tmp_path):
        path = tmp_path / "gt.npz"
        np.savez(path, gt_poses=np.zeros((up1.N_PAIRS_TOTAL, 6), dtype=np.float64))
        poses, lineage = up1.load_gt_poses(path)
        assert lineage == "av_pyav"
        assert poses.shape == (up1.N_PAIRS_TOTAL, 6)

    def test_dali_pt_reports_the_dali_lineage(self, tmp_path):
        torch = pytest.importorskip("torch")
        path = tmp_path / "gt_cache_dali.pt"
        torch.save({"pose": torch.zeros(up1.N_PAIRS_TOTAL, 6)}, path)
        poses, lineage = up1.load_gt_poses(path)
        assert lineage == "dali"
        assert poses.shape == (up1.N_PAIRS_TOTAL, 6)

    def test_refuses_a_pt_without_pose(self, tmp_path):
        torch = pytest.importorskip("torch")
        path = tmp_path / "gt_cache_dali.pt"
        torch.save({"seg": torch.zeros(2)}, path)
        with pytest.raises(up1.Up1Error, match="no 'pose' key"):
            up1.load_gt_poses(path)

    def test_refuses_wrong_pose_shape(self, tmp_path):
        path = tmp_path / "gt.npz"
        np.savez(path, gt_poses=np.zeros((7, 6), dtype=np.float64))
        with pytest.raises(up1.Up1Error, match="expected"):
            up1.load_gt_poses(path)

    def test_refuses_a_missing_cache(self, tmp_path):
        with pytest.raises(up1.Up1Error, match="does not exist"):
            up1.load_gt_poses(tmp_path / "absent.npz")


class TestApplyOffset:
    def test_zero_offset_is_identity(self):
        frames = np.array([[0, 7, 128, 255]], dtype=np.uint8)
        out = up1.apply_offset(frames, 0.0, realize_uint8=True)
        assert np.array_equal(out, frames.astype(np.float32))

    def test_realized_offset_clips_to_the_uint8_lattice(self):
        frames = np.array([[0, 254, 255]], dtype=np.uint8)
        out = up1.apply_offset(frames, 3.0, realize_uint8=True)
        assert out.max() <= 255.0
        assert np.array_equal(out, np.array([[3.0, 255.0, 255.0]], dtype=np.float32))

    def test_negative_offset_clips_at_zero(self):
        frames = np.array([[0, 1, 10]], dtype=np.uint8)
        out = up1.apply_offset(frames, -5.0, realize_uint8=True)
        assert out.min() >= 0.0
        assert np.array_equal(out, np.array([[0.0, 0.0, 5.0]], dtype=np.float32))

    def test_float_mode_keeps_sub_lsb_detail(self):
        frames = np.array([[100]], dtype=np.uint8)
        out = up1.apply_offset(frames, -0.2, realize_uint8=False)
        assert out[0, 0] == pytest.approx(99.8, abs=1e-5)

    def test_realized_mode_rounds_away_sub_lsb_detail(self):
        frames = np.array([[100]], dtype=np.uint8)
        out = up1.apply_offset(frames, -0.2, realize_uint8=True)
        assert out[0, 0] == pytest.approx(100.0)

    def test_does_not_mutate_the_input(self):
        frames = np.array([[10, 20]], dtype=np.uint8)
        before = frames.copy()
        up1.apply_offset(frames, 5.0, realize_uint8=True)
        assert np.array_equal(frames, before)


class TestCheckpoint:
    def test_round_trips_and_resumes(self, tmp_path):
        path = tmp_path / "rows.jsonl"
        state = up1.Checkpoint.load(path)
        assert not state.has(0.0, 3)
        state.append({"offset": 0.0, "pair": 3, "d_pose": 1.0})
        assert state.has(0.0, 3)
        # A fresh load must see the persisted row: resumability is P0.
        assert up1.Checkpoint.load(path).has(0.0, 3)

    def test_distinguishes_offsets_for_the_same_pair(self, tmp_path):
        state = up1.Checkpoint.load(tmp_path / "rows.jsonl")
        state.append({"offset": 0.0, "pair": 3, "d_pose": 1.0})
        assert not state.has(1.0, 3)
