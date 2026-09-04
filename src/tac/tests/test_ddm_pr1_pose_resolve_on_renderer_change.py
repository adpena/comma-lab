"""Tests for ``experiments/ddm_pr1_pose_resolve_on_renderer_change``.

The load-bearing guards here are the ones that would let a WRONG number leave
the arm silently:

* ``RenderedOddFrames`` must refuse EVEN indices.  Frame 0 is the pose carrier's
  and is re-rendered per candidate code block; if the odd-frame adapter ever
  served an even index the solver would optimise frame 0 against a copy of
  itself and every recovery number would be fiction.
* ``select_pairs`` must never return a prefix.  A contiguous prefix of this video
  measures the POSE axis 2.54-4.21x harder than the population, and the pose axis
  is the entire output of this arm.
* ``payable_pose_ceiling`` must invert the AFR1 promotion arithmetic exactly, and
  must not manufacture a budget out of a seg REGRESSION.
* ``build_instrument`` must refuse a body whose sha256 is not the one declared,
  and must refuse a body carrying a compensation overlay (a path this arm has
  not measured).
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
_MODULE_PATH = REPO / "experiments" / "ddm_pr1_pose_resolve_on_renderer_change.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("ddm_pr1_undertest", _MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


pr1 = _load_module()


class _FakeRenderedOddFrames(pr1.RenderedOddFrames):
    """``RenderedOddFrames`` index arithmetic without loading a renderer."""

    def __init__(self, n_pairs: int = pr1.N_PAIRS):
        self._tokens = None
        self._cache = {}
        self._cache_pairs = 4
        self.renders = 0
        self._n = n_pairs

    def _render(self, pair: int) -> np.ndarray:  # pragma: no cover - trivial
        self.renders += 1
        return np.full((2, 2, 3), pair % 256, dtype=np.uint8)


# ---------------------------------------------------------------------------
# The odd-frame adapter
# ---------------------------------------------------------------------------


class TestRenderedOddFrames:
    def test_serves_odd_indices_as_pairs(self):
        raw = _FakeRenderedOddFrames()
        out = raw[np.array([1, 3, 1199])]
        assert out.shape == (3, 2, 2, 3)
        assert out[0, 0, 0, 0] == 0
        assert out[1, 0, 0, 0] == 1
        assert out[2, 0, 0, 0] == 599 % 256

    def test_refuses_even_indices(self):
        raw = _FakeRenderedOddFrames()
        with pytest.raises(pr1.Pr1Error, match="ODD"):
            raw[np.array([0])]
        with pytest.raises(pr1.Pr1Error, match="ODD"):
            raw[np.array([1, 2])]

    def test_refuses_out_of_range_pairs(self):
        raw = _FakeRenderedOddFrames()
        with pytest.raises(pr1.Pr1Error, match="out of range"):
            raw[np.array([2 * pr1.N_PAIRS + 1])]

    def test_matches_the_solver_indexing_convention(self):
        """``2 * pair + 1`` is exactly what br1/up2/jg5 ask for."""
        raw = _FakeRenderedOddFrames()
        pair = np.array([137], dtype=np.int64)
        assert raw[2 * pair + 1][0, 0, 0, 0] == 137

    def test_caches_and_evicts_without_changing_values(self):
        raw = _FakeRenderedOddFrames()
        first = raw[np.array([1])].copy()
        assert raw.renders == 1
        assert np.array_equal(raw[np.array([1])], first)
        assert raw.renders == 1, "a cache hit must not re-render"
        raw[np.array([3, 5, 7, 9, 11])]
        assert np.array_equal(raw[np.array([1])], first), "eviction must not change values"

    def test_field_sha256_is_deterministic_and_order_sensitive(self):
        raw = _FakeRenderedOddFrames()
        pairs = np.array([0, 1, 2], dtype=np.int64)
        assert raw.field_sha256(pairs) == raw.field_sha256(pairs)
        assert raw.field_sha256(pairs) != raw.field_sha256(pairs[::-1])


# ---------------------------------------------------------------------------
# Pair selection and sharding
# ---------------------------------------------------------------------------


class TestPairSelection:
    def test_full_field_is_the_whole_population(self):
        assert np.array_equal(pr1.select_pairs(600, 1), np.arange(600))
        assert np.array_equal(pr1.select_pairs(9999, 1), np.arange(600))

    def test_subset_is_not_a_prefix(self):
        pairs = pr1.select_pairs(200, 20260904)
        assert len(pairs) == 200
        assert not np.array_equal(pairs, np.arange(200)), "a prefix is the false-negative shape"
        assert pairs.max() > 400, "a subset must reach the back of the video"
        assert len(np.unique(pairs)) == 200

    def test_subset_is_seeded_and_reproducible(self):
        assert np.array_equal(pr1.select_pairs(50, 7), pr1.select_pairs(50, 7))
        assert not np.array_equal(pr1.select_pairs(50, 7), pr1.select_pairs(50, 8))

    def test_refuses_empty_selection(self):
        with pytest.raises(pr1.Pr1Error):
            pr1.select_pairs(0, 1)

    def test_shard_is_strided_not_contiguous(self):
        pairs = np.arange(600)
        shard = pr1.shard_of(pairs, 0, 2)
        assert len(shard) == 300
        assert shard[0] == 0 and shard[1] == 2, "a shard must stride, never block"
        union = np.sort(np.concatenate([pr1.shard_of(pairs, i, 2) for i in range(2)]))
        assert np.array_equal(union, pairs), "shards must partition the field"

    def test_shard_bounds_are_checked(self):
        with pytest.raises(pr1.Pr1Error):
            pr1.shard_of(np.arange(600), 2, 2)
        with pytest.raises(pr1.Pr1Error):
            pr1.shard_of(np.arange(600), 0, 0)


# ---------------------------------------------------------------------------
# The promotion arithmetic
# ---------------------------------------------------------------------------


class TestPayablePoseCeiling:
    def test_inverts_the_promotion_inequality_exactly(self):
        delta = -5.0348e-05  # a 25% cut of the AFR1 T4 d_seg
        ceiling = pr1.payable_pose_ceiling(delta)
        assert math.isclose(
            math.sqrt(10.0 * ceiling),
            pr1.AFR1_POSE_LEG_T4 + 100.0 * abs(delta),
            rel_tol=1e-12,
        )

    def test_reproduces_ft1_derived_ceilings(self):
        """ft1 Sec 7's table, re-derived rather than copied."""
        assert pr1.payable_pose_ceiling(-0.25 * pr1.AFR1_D_SEG_T4) == pytest.approx(
            1.694e-05, rel=2e-3
        )
        assert pr1.payable_pose_ceiling(-0.10 * pr1.AFR1_D_SEG_T4) == pytest.approx(
            9.99e-06, rel=2e-3
        )

    def test_a_seg_regression_buys_no_pose_budget(self):
        base = pr1.payable_pose_ceiling(0.0)
        assert pr1.payable_pose_ceiling(+1e-4) == pytest.approx(base)
        assert base == pytest.approx(pr1.AFR1_POSE_LEG_T4**2 / 10.0, rel=1e-12)

    def test_ceiling_is_monotone_in_the_seg_cut(self):
        cuts = [-1e-6, -1e-5, -5e-5, -1e-4]
        ceilings = [pr1.payable_pose_ceiling(c) for c in cuts]
        assert ceilings == sorted(ceilings)


class TestScoreArithmetic:
    def test_composed_score_reproduces_the_afr1_receipt(self):
        got = pr1.composed_score(
            pr1.AFR1_D_SEG_T4, pr1.AFR1_D_POSE_T4, pr1.FRONTIER_ARCHIVE_BYTES
        )
        assert got == pytest.approx(pr1.AFR1_SCORE_T4, rel=1e-6)

    def test_pose_leg_matches_the_receipt_leg(self):
        assert pr1.pose_leg(pr1.AFR1_D_POSE_T4) == pytest.approx(
            pr1.AFR1_POSE_LEG_T4, rel=1e-5
        )


# ---------------------------------------------------------------------------
# Fail-closed gates
# ---------------------------------------------------------------------------


class TestInstrumentGates:
    def test_refuses_a_body_whose_sha_is_not_the_declared_one(self, tmp_path):
        runtime = tmp_path / "runtime"
        runtime.mkdir()
        (runtime / "archive.zip").write_bytes(b"not the frontier body")
        with pytest.raises(pr1.Pr1Error, match="unidentified body"):
            pr1.build_instrument(
                runtime=runtime, gt_cache=tmp_path / "gt.pt", axis="contest_cuda",
                renderer_source=tmp_path / "r.bin", tokens_path=tmp_path / "t.u8",
            )

    def test_declared_frontier_constants_are_the_afr1_body(self):
        assert pr1.FRONTIER_ARCHIVE_BYTES == 180_002
        assert pr1.FRONTIER_ARCHIVE_SHA256.startswith("cbb8d928")

    def test_token_field_shape_is_enforced(self, tmp_path):
        short = tmp_path / "tokens.u8"
        short.write_bytes(b"\x00" * 16)
        with pytest.raises(pr1.Pr1Error, match="token field"):
            pr1.load_tokens(short)


class TestPayloadDigests:
    def test_array_digest_is_value_sensitive(self):
        a = np.arange(12, dtype=np.int32)
        b = a.copy()
        assert pr1.sha256_array(a) == pr1.sha256_array(b)
        b[3] += 1
        assert pr1.sha256_array(a) != pr1.sha256_array(b)

    def test_array_digest_is_layout_stable(self):
        a = np.arange(12, dtype=np.int32).reshape(3, 4)
        assert pr1.sha256_array(a) == pr1.sha256_array(np.ascontiguousarray(a[:, :]))

    def test_file_digest_matches_hashlib(self, tmp_path):
        import hashlib

        path = tmp_path / "blob.bin"
        payload = b"pr1" * 1000
        path.write_bytes(payload)
        assert pr1.sha256_file(path) == hashlib.sha256(payload).hexdigest()


class TestSolverSelection:
    def test_default_solver_is_the_optimal_form_not_the_truncated_one(self):
        args = pr1.build_parser().parse_args(
            ["solve", "--runtime", "/r", "--gt-cache", "/g", "--renderer", "/c",
             "--tokens", "/t", "--out", "/o"]
        )
        assert args.solver == "jg5", (
            "up2's +-2 radius is a measured truncation (jg5 Sec 4); defaulting to it "
            "would report the solver's weakness as the carrier's ceiling"
        )

    def test_materiality_floor_defaults_to_the_target_operating_point(self):
        args = pr1.build_parser().parse_args(
            ["solve", "--runtime", "/r", "--gt-cache", "/g", "--renderer", "/c",
             "--tokens", "/t", "--out", "/o"]
        )
        assert args.materiality_operating_point == pr1.AFR1_D_POSE_T4

    def test_measure_defaults_to_one_declared_batch_shape(self):
        args = pr1.build_parser().parse_args(
            ["measure", "--runtime", "/r", "--gt-cache", "/g", "--renderer", "/c",
             "--tokens", "/t", "--out", "/o"]
        )
        assert args.batch_size == pr1.DEFAULT_BATCH
        assert args.pairs == pr1.N_PAIRS


class TestResumability:
    def test_rows_resume_from_disk_and_last_row_wins(self, tmp_path):
        rows = tmp_path / "rows.jsonl"
        rows.write_text(
            '{"pair": 3, "final_d_pose": 1.0}\n'
            "\n"
            "not json\n"
            '{"pair": 3, "final_d_pose": 0.5}\n'
            '{"pair": 9, "final_d_pose": 2.0}\n',
            encoding="utf-8",
        )
        done = pr1.load_done(rows)
        assert set(done) == {3, 9}
        assert done[3]["final_d_pose"] == 0.5

    def test_missing_rows_file_is_an_empty_resume(self, tmp_path):
        assert pr1.load_done(tmp_path / "absent.jsonl") == {}


class TestRenderDigestIsOptOut:
    def test_measure_does_not_pay_the_digest_pass_by_default(self):
        args = pr1.build_parser().parse_args(
            ["measure", "--runtime", "/r", "--gt-cache", "/g", "--renderer", "/c",
             "--tokens", "/t", "--out", "/o"]
        )
        assert args.render_digest is False

    def test_measure_accepts_the_digest_flag(self):
        args = pr1.build_parser().parse_args(
            ["measure", "--runtime", "/r", "--gt-cache", "/g", "--renderer", "/c",
             "--tokens", "/t", "--out", "/o", "--render-digest"]
        )
        assert args.render_digest is True


class TestCodesMergeGate:
    def test_merge_refuses_a_foreign_body(self, tmp_path):
        runtime = tmp_path / "runtime"
        runtime.mkdir()
        (runtime / "archive.zip").write_bytes(b"some other archive")
        rows = tmp_path / "rows.jsonl"
        rows.write_text('{"pair": 1, "codes": [0]}\n', encoding="utf-8")
        args = pr1.build_parser().parse_args(
            ["codes", "--runtime", str(runtime), "--rows", str(rows),
             "--out", str(tmp_path / "codes.npy")]
        )
        with pytest.raises(pr1.Pr1Error, match="another body"):
            pr1.run_codes(args)

    def test_merge_takes_the_last_row_per_pair_across_shards(self, tmp_path):
        a = tmp_path / "a.jsonl"
        b = tmp_path / "b.jsonl"
        a.write_text('{"pair": 0, "codes": [1, 1]}\n', encoding="utf-8")
        b.write_text('{"pair": 1, "codes": [2, 2]}\n', encoding="utf-8")
        merged = {}
        for path in (a, b):
            merged.update(pr1.load_done(path))
        assert merged[0]["codes"] == [1, 1]
        assert merged[1]["codes"] == [2, 2]


def _measure_stub(tmp_path, name, values, *, batch_size=8, pairs_index=(0, 1, 2),
                  gt="gt.pt", sha=pr1.FRONTIER_ARCHIVE_SHA256):
    import json

    arr = np.asarray(values, dtype=np.float64)
    payload_dir = tmp_path / f"{name}_payload"
    payload_dir.mkdir(parents=True, exist_ok=True)
    arr_path = payload_dir / "per_pair_d_pose.npy"
    np.save(arr_path, arr)
    doc = {
        "schema": "tac.ddm_pr1.measure.v1",
        "batch_size": batch_size,
        "pairs_index": list(pairs_index),
        "pair_selection": "full n600",
        "instrument": {"gt_cache": gt, "archive_sha256": sha},
        "payload": {"per_pair_d_pose": {"path": str(arr_path)}},
    }
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


class TestReportGuards:
    def test_refuses_cross_instrument_differencing(self, tmp_path):
        a = _measure_stub(tmp_path, "a", [1.0, 1.0, 1.0])
        b = _measure_stub(tmp_path, "b", [1.0, 1.0, 1.0], batch_size=32)
        c = _measure_stub(tmp_path, "c", [1.0, 1.0, 1.0])
        args = pr1.build_parser().parse_args(
            ["report", "--base-measure", str(a), "--before-measure", str(b),
             "--after-measure", str(c), "--delta-d-seg", "1e-5",
             "--delta-d-seg-source", "test", "--out", str(tmp_path / "r.json")]
        )
        with pytest.raises(pr1.Pr1Error, match="cross-instrument"):
            pr1.run_report(args)

    def test_refuses_a_different_pair_set(self, tmp_path):
        a = _measure_stub(tmp_path, "a", [1.0, 1.0, 1.0])
        b = _measure_stub(tmp_path, "b", [1.0, 1.0, 1.0], pairs_index=(0, 1, 5))
        c = _measure_stub(tmp_path, "c", [1.0, 1.0, 1.0])
        args = pr1.build_parser().parse_args(
            ["report", "--base-measure", str(a), "--before-measure", str(b),
             "--after-measure", str(c), "--delta-d-seg", "1e-5",
             "--delta-d-seg-source", "test", "--out", str(tmp_path / "r.json")]
        )
        with pytest.raises(pr1.Pr1Error, match="cross-instrument"):
            pr1.run_report(args)

    def test_refuses_a_foreign_schema(self, tmp_path):
        import json

        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"schema": "something.else"}), encoding="utf-8")
        with pytest.raises(pr1.Pr1Error, match="not a ddm_pr1 measure"):
            pr1._load_measure(bad)

    def test_coupling_and_payability_are_reported_apart(self, tmp_path):
        base = _measure_stub(tmp_path, "base", [1e-6, 1e-6, 1e-6])
        before = _measure_stub(tmp_path, "before", [1e-2, 1e-2, 1e-2])
        after = _measure_stub(tmp_path, "after", [2e-6, 2e-6, 2e-6])
        out = tmp_path / "r.json"
        args = pr1.build_parser().parse_args(
            ["report", "--base-measure", str(base), "--before-measure", str(before),
             "--after-measure", str(after), "--delta-d-seg", "6.9e-5",
             "--delta-d-seg-source", "test", "--out", str(out)]
        )
        assert pr1.run_report(args) == 0
        import json

        got = json.loads(out.read_text(encoding="utf-8"))
        assert got["recovery"]["mean_based"] == pytest.approx(5000.0)
        assert got["coupling"]["post_re_solve"] == pytest.approx(
            (2e-6 - 1e-6) / 6.9e-5
        )
        assert got["coupling"]["pre_re_solve"] == pytest.approx(
            (1e-2 - 1e-6) / 6.9e-5
        )
        # the two verdicts are distinct fields, never one conflated flag
        assert "prediction_holds" in got["charter_prediction"]
        assert "payable" in got["closing_arithmetic"]
        assert got["closing_arithmetic"]["k_post_payable_bar"] == pytest.approx(
            (pr1.payable_pose_ceiling(-0.25 * pr1.AFR1_D_SEG_T4) - pr1.AFR1_D_POSE_T4)
            / (0.25 * pr1.AFR1_D_SEG_T4)
        )

    def test_recovery_quantiles_expose_the_tail(self, tmp_path):
        base = _measure_stub(tmp_path, "base", [1e-6] * 3)
        before = _measure_stub(tmp_path, "before", [1e-2, 1e-2, 1e-2])
        after = _measure_stub(tmp_path, "after", [1e-9, 1e-3, 1e-6])
        out = tmp_path / "r.json"
        args = pr1.build_parser().parse_args(
            ["report", "--base-measure", str(base), "--before-measure", str(before),
             "--after-measure", str(after), "--delta-d-seg", "6.9e-5",
             "--delta-d-seg-source", "test", "--out", str(out)]
        )
        pr1.run_report(args)
        import json

        got = json.loads(out.read_text(encoding="utf-8"))
        rec = got["recovery"]
        assert rec["median_per_pair"] == pytest.approx(1e4)
        assert rec["mean_based"] < rec["median_per_pair"], (
            "the mean is a mean of per-pair MSEs; the worst pairs own it"
        )
        assert rec["pairs_improved"] == 3
