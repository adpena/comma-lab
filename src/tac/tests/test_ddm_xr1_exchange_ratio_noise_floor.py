"""Tests for ``experiments/ddm_xr1_exchange_ratio_noise_floor.py``.

These cover the parts that decide whether the REPORTED INTERVAL IS REAL:

* the score arithmetic (a wrong ``sqrt(10*d_pose)`` silently rescales every
  pose claim the memo makes);
* the two calibrated bootstraps (the fixed adjustment must absorb exactly the
  ideal-vs-exact residual, and the identity draw must reproduce the retained
  exact total -- otherwise the interval is centred on a number we never
  physically measured);
* the per-pair scorer parse (a duplicated or missing pair silently changes the
  denominator of every mean);
* the RN1 gradability probe, whose whole job is to REFUSE the substring trap:
  ``d_seg_per_pair_max`` is a scalar summary, and a probe that counts it as a
  per-pair vector reports gradable rows that do not exist;
* the physical-repeat summary, which must RECORD a non-identical repeat rather
  than raise it away -- the falsifier this arm exists to expose.

They do NOT re-run the RC64 coder or any scorer.  Those are the shipped
runtime's own code, imported rather than reimplemented, and the module proves
them empirically with retained physical receipts instead of by unit test.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO / "experiments" / "ddm_xr1_exchange_ratio_noise_floor.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "ddm_xr1_exchange_ratio_noise_floor", MODULE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["ddm_xr1_exchange_ratio_noise_floor"] = module
    spec.loader.exec_module(module)
    return module


xr1 = _load_module()


def _draws(rows: int, *, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, xr1.PAIR_COUNT, size=(rows, xr1.PAIR_COUNT), dtype=np.uint16)


def _identity_draw() -> np.ndarray:
    return np.arange(xr1.PAIR_COUNT, dtype=np.uint16)[None, :]


# --------------------------------------------------------------------------
# score arithmetic
# --------------------------------------------------------------------------


def test_score_delta_matches_the_contest_formula_term_by_term():
    result = xr1.score_delta(
        base_d_seg=0.002,
        candidate_d_seg=0.0021,
        base_d_pose=1e-5,
        candidate_d_pose=1.1e-5,
        delta_bytes=-2_940,
    )
    assert result["delta_s_seg"] == pytest.approx(100.0 * (0.0021 - 0.002))
    assert result["delta_s_pose"] == pytest.approx(
        math.sqrt(10.0 * 1.1e-5) - math.sqrt(10.0 * 1e-5)
    )
    assert result["delta_s_rate"] == pytest.approx(25.0 * -2_940 / 37_545_489)
    assert result["delta_s"] == pytest.approx(
        result["delta_s_seg"] + result["delta_s_pose"] + result["delta_s_rate"]
    )
    assert result["exchange_ratio"] == pytest.approx(
        result["delta_s_rate"] / result["delta_s_distortion"]
    )


def test_score_delta_pose_term_is_nonlinear_not_a_scaled_difference():
    """sqrt(10*d) is concave: equal d_pose steps must give unequal S steps."""
    low = xr1.score_delta(
        base_d_seg=0.0,
        candidate_d_seg=0.0,
        base_d_pose=1e-6,
        candidate_d_pose=2e-6,
        delta_bytes=0,
    )
    high = xr1.score_delta(
        base_d_seg=0.0,
        candidate_d_seg=0.0,
        base_d_pose=1e-4,
        candidate_d_pose=1.01e-4,
        delta_bytes=0,
    )
    assert low["delta_s_pose"] != pytest.approx(high["delta_s_pose"])
    assert low["delta_s_pose"] > 0.0


def test_rate_denominator_is_the_contest_source_size():
    assert xr1.RATE_DENOMINATOR_BYTES == 37_545_489
    assert xr1.RATE_NUMERATOR == 25.0
    assert xr1.PAIR_COUNT == 600


# --------------------------------------------------------------------------
# calibrated bootstraps
# --------------------------------------------------------------------------


def test_total_bootstrap_identity_draw_reproduces_the_exact_retained_delta():
    base = np.full(xr1.PAIR_COUNT, 1_000.0)
    candidate = base - 8.0  # -1 byte of ideal codelength per pair
    _, samples, fixed = xr1.exact_total_calibrated_bootstrap(
        base, candidate, _identity_draw(), exact_delta_bytes=-597
    )
    # ideal total is -600 B; the retained physical delta is -597 B.
    assert fixed == pytest.approx(3.0)
    assert samples[0] == pytest.approx(-597.0)


def test_total_bootstrap_pair_delta_is_bits_over_eight():
    base = np.zeros(xr1.PAIR_COUNT)
    candidate = np.zeros(xr1.PAIR_COUNT)
    candidate[0] = -16.0
    pair_delta, _, _ = xr1.exact_total_calibrated_bootstrap(
        base, candidate, _identity_draw(), exact_delta_bytes=-2
    )
    assert pair_delta[0] == pytest.approx(-2.0)
    assert pair_delta[1:].sum() == pytest.approx(0.0)


def test_total_bootstrap_actually_varies_across_resamples():
    rng = np.random.default_rng(3)
    base = np.zeros(xr1.PAIR_COUNT)
    candidate = rng.normal(-8.0, 40.0, size=xr1.PAIR_COUNT)
    _, samples, _ = xr1.exact_total_calibrated_bootstrap(
        base, candidate, _draws(50), exact_delta_bytes=-600
    )
    assert samples.shape == (50,)
    assert float(np.std(samples)) > 0.0


def test_total_bootstrap_refuses_a_ledger_that_is_not_n600():
    short = np.zeros(xr1.PAIR_COUNT - 1)
    with pytest.raises(xr1.Xr1Error, match="n600"):
        xr1.exact_total_calibrated_bootstrap(
            short, short, _identity_draw(), exact_delta_bytes=0
        )


def test_total_bootstrap_refuses_a_draw_matrix_of_the_wrong_width():
    base = np.zeros(xr1.PAIR_COUNT)
    bad = np.zeros((4, xr1.PAIR_COUNT - 1), dtype=np.uint16)
    with pytest.raises(xr1.Xr1Error, match="600 pair indices"):
        xr1.exact_total_calibrated_bootstrap(base, base, bad, exact_delta_bytes=0)


def test_mean_bootstrap_identity_draw_reproduces_the_retained_aggregate():
    values = np.linspace(0.001, 0.003, xr1.PAIR_COUNT)
    samples, fixed = xr1.exact_mean_calibrated_bootstrap(
        values, _identity_draw(), exact_mean=0.00201
    )
    assert samples[0] == pytest.approx(0.00201)
    assert fixed == pytest.approx(0.00201 - float(values.mean()))


def test_mean_bootstrap_refuses_a_vector_that_is_not_n600():
    with pytest.raises(xr1.Xr1Error, match="n600"):
        xr1.exact_mean_calibrated_bootstrap(
            np.zeros(10), _identity_draw(), exact_mean=0.0
        )


def test_percentile_interval_is_the_symmetric_95_percent_band():
    values = np.arange(0.0, 1001.0)
    interval = xr1.percentile_interval(values)
    assert interval["low"] == pytest.approx(25.0)
    assert interval["high"] == pytest.approx(975.0)
    assert interval["width"] == pytest.approx(950.0)
    assert interval["half_width"] == pytest.approx(475.0)


# --------------------------------------------------------------------------
# per-pair scorer parse
# --------------------------------------------------------------------------


def _receipt(stages):
    return {"pair_count": xr1.PAIR_COUNT, "batch_stages": stages}


def _full_stage():
    return [
        {
            "pair_start": 0,
            "pair_stop_exclusive": xr1.PAIR_COUNT,
            "d_seg_per_pair": [0.001] * xr1.PAIR_COUNT,
            "d_pose_per_pair": [1e-5] * xr1.PAIR_COUNT,
        }
    ]


def test_scorer_pair_vectors_reads_a_complete_population():
    seg, pose = xr1.scorer_pair_vectors(_receipt(_full_stage()))
    assert seg.shape == (xr1.PAIR_COUNT,)
    assert pose.shape == (xr1.PAIR_COUNT,)
    assert seg.mean() == pytest.approx(0.001)


def test_scorer_pair_vectors_refuses_a_duplicated_pair():
    stages = [
        {
            "pair_start": 0,
            "pair_stop_exclusive": 300,
            "d_seg_per_pair": [0.0] * 300,
            "d_pose_per_pair": [0.0] * 300,
        },
        {
            "pair_start": 0,
            "pair_stop_exclusive": 300,
            "d_seg_per_pair": [0.0] * 300,
            "d_pose_per_pair": [0.0] * 300,
        },
    ]
    with pytest.raises(xr1.Xr1Error, match="duplicate"):
        xr1.scorer_pair_vectors(_receipt(stages))


def test_scorer_pair_vectors_refuses_a_partial_population():
    stages = [
        {
            "pair_start": 0,
            "pair_stop_exclusive": 300,
            "d_seg_per_pair": [0.0] * 300,
            "d_pose_per_pair": [0.0] * 300,
        }
    ]
    with pytest.raises(xr1.Xr1Error, match="0\\.\\.599"):
        xr1.scorer_pair_vectors(_receipt(stages))


def test_scorer_pair_vectors_refuses_a_receipt_that_is_not_n600():
    payload = _receipt(_full_stage())
    payload["pair_count"] = 96
    with pytest.raises(xr1.Xr1Error, match="not n600"):
        xr1.scorer_pair_vectors(payload)


def test_scorer_pair_vectors_refuses_a_stage_whose_vector_length_drifted():
    stages = _full_stage()
    stages[0]["d_pose_per_pair"] = [0.0] * (xr1.PAIR_COUNT - 1)
    with pytest.raises(xr1.Xr1Error, match="length drifted"):
        xr1.scorer_pair_vectors(_receipt(stages))


# --------------------------------------------------------------------------
# RN1 gradability probe
# --------------------------------------------------------------------------


def test_per_pair_vector_lengths_ignores_the_scalar_max_summary():
    """The substring trap: `d_seg_per_pair_max` is a scalar, not a vector."""
    blob = {"d_seg_per_pair_max": 0.42, "nested": {"d_seg_per_pair_max": 0.9}}
    assert xr1._per_pair_vector_lengths(blob, "d_seg_per_pair") == []


def test_per_pair_vector_lengths_finds_exact_keys_at_any_depth():
    blob = {"stages": [{"d_seg_per_pair": [0.0] * 300}, {"d_seg_per_pair": [0.0] * 300}]}
    assert sum(xr1._per_pair_vector_lengths(blob, "d_seg_per_pair")) == 600


def test_arm_store_prefix_takes_the_ddm_slug():
    assert (
        xr1.arm_store_prefix(".omx/research/ddm_fs3_jg5_real_price_reopen_20260820.md")
        == "ddm_fs3"
    )
    assert (
        xr1.arm_store_prefix(".omx/research/ddm_pa1r_pool_a_race_20260730.md")
        == "ddm_pa1r"
    )


def test_probe_finds_an_n600_byte_ledger_and_ignores_a_wrong_shape_one(tmp_path):
    np.save(tmp_path / "bits_per_frame_good.npy", np.zeros(xr1.PAIR_COUNT))
    np.save(tmp_path / "bits_per_frame_n96.npy", np.zeros(96))
    probe = xr1.probe_per_pair_custody(tmp_path)
    assert len(probe["n600_byte_ledgers"]) == 1
    assert probe["n600_byte_ledgers"][0].endswith("bits_per_frame_good.npy")


def test_probe_ignores_a_scalar_max_receipt_but_finds_a_real_vector(tmp_path):
    (tmp_path / "scalar.json").write_text(json.dumps({"d_seg_per_pair_max": 0.4}))
    (tmp_path / "vector.json").write_text(
        json.dumps({"d_seg_per_pair": [0.0] * xr1.PAIR_COUNT})
    )
    probe = xr1.probe_per_pair_custody(tmp_path)
    assert len(probe["n600_d_seg_receipts"]) == 1
    assert probe["n600_d_seg_receipts"][0].endswith("vector.json")
    assert probe["n600_d_pose_receipts"] == []


def test_grade_row_custody_reports_no_store_when_nothing_matches(tmp_path, monkeypatch):
    monkeypatch.setattr(xr1, "GRADABILITY_SEARCH_ROOTS", (tmp_path,))
    verdict = xr1.grade_row_custody(".omx/research/ddm_zzz_nothing_20260101.md")
    assert verdict["grade"] == "UNGRADABLE_NO_STORE"
    assert verdict["stores_found"] == []


def test_grade_row_custody_is_rate_only_without_a_distortion_receipt(
    tmp_path, monkeypatch
):
    store = tmp_path / "ddm_zzz_store"
    store.mkdir()
    np.save(store / "bits_per_frame_x.npy", np.zeros(xr1.PAIR_COUNT))
    monkeypatch.setattr(xr1, "GRADABILITY_SEARCH_ROOTS", (tmp_path,))
    verdict = xr1.grade_row_custody(".omx/research/ddm_zzz_thing_20260101.md")
    assert verdict["grade"] == "UNGRADABLE_RATE_ONLY"
    assert verdict["has_n600_byte_ledger"] is True
    assert verdict["has_n600_d_seg"] is False


def test_grade_row_custody_is_distortion_only_without_a_byte_ledger(
    tmp_path, monkeypatch
):
    store = tmp_path / "ddm_zzz_store"
    store.mkdir()
    (store / "seg.json").write_text(
        json.dumps({"d_seg_per_pair": [0.0] * xr1.PAIR_COUNT})
    )
    monkeypatch.setattr(xr1, "GRADABILITY_SEARCH_ROOTS", (tmp_path,))
    verdict = xr1.grade_row_custody(".omx/research/ddm_zzz_thing_20260101.md")
    assert verdict["grade"] == "UNGRADABLE_DISTORTION_ONLY"


def test_grade_row_custody_is_gradable_only_with_all_three_receipts(
    tmp_path, monkeypatch
):
    store = tmp_path / "ddm_zzz_store"
    store.mkdir()
    np.save(store / "bits_per_frame_x.npy", np.zeros(xr1.PAIR_COUNT))
    (store / "d.json").write_text(
        json.dumps(
            {
                "d_seg_per_pair": [0.0] * xr1.PAIR_COUNT,
                "d_pose_per_pair": [0.0] * xr1.PAIR_COUNT,
            }
        )
    )
    monkeypatch.setattr(xr1, "GRADABILITY_SEARCH_ROOTS", (tmp_path,))
    verdict = xr1.grade_row_custody(".omx/research/ddm_zzz_thing_20260101.md")
    assert verdict["grade"] == "GRADABLE"


def test_grade_row_custody_reports_empty_store_as_no_per_pair_data(
    tmp_path, monkeypatch
):
    (tmp_path / "ddm_zzz_store").mkdir()
    monkeypatch.setattr(xr1, "GRADABILITY_SEARCH_ROOTS", (tmp_path,))
    verdict = xr1.grade_row_custody(".omx/research/ddm_zzz_thing_20260101.md")
    assert verdict["grade"] == "UNGRADABLE_NO_PER_PAIR_DATA"


# --------------------------------------------------------------------------
# physical repeat summary -- the falsifier must survive
# --------------------------------------------------------------------------


def _repeat_row(byte_count: int, *, identical: bool):
    return {
        "stream": {"bytes": byte_count},
        "stream_vs_repeat_0": {"byte_identical": identical},
        "archive_vs_repeat_0": {"byte_identical": identical},
    }


def test_physical_summary_reports_sigma_zero_when_the_prior_law_holds():
    rows = [_repeat_row(180_002, identical=True) for _ in range(3)]
    payload = xr1.summarize_physical_repeats(rows)
    assert payload["sigma_b_sample_bytes"] == 0.0
    assert payload["spread_max_minus_min_bytes"] == 0
    assert payload["all_streams_byte_identical"] is True
    assert payload["prior_law_prediction_held"] is True


def test_physical_summary_records_a_non_identical_repeat_instead_of_raising():
    """The falsifier must reach the receipt; an arm may not delete its own refutation."""
    rows = [
        _repeat_row(180_002, identical=True),
        _repeat_row(180_009, identical=False),
        _repeat_row(180_002, identical=True),
    ]
    payload = xr1.summarize_physical_repeats(rows)
    assert payload["all_streams_byte_identical"] is False
    assert payload["prior_law_prediction_held"] is False
    assert payload["spread_max_minus_min_bytes"] == 7
    assert payload["sigma_b_sample_bytes"] > 0.0
    assert payload["stream_byte_identical_per_repeat"] == [True, False, True]


def test_physical_summary_flags_identical_lengths_that_are_not_identical_bytes():
    """Equal byte COUNTS with differing bytes still refutes determinism."""
    rows = [_repeat_row(180_002, identical=True), _repeat_row(180_002, identical=False)]
    payload = xr1.summarize_physical_repeats(rows)
    assert payload["sigma_b_sample_bytes"] == 0.0
    assert payload["all_streams_byte_identical"] is False
    assert payload["prior_law_prediction_held"] is False


def test_physical_summary_refuses_an_empty_repeat_set():
    with pytest.raises(xr1.Xr1Error, match="at least one"):
        xr1.summarize_physical_repeats([])


# --------------------------------------------------------------------------
# module-level invariants
# --------------------------------------------------------------------------


def test_bootstrap_spec_matches_the_charter():
    assert xr1.BOOTSTRAP_RESAMPLES == 200
    assert xr1.PHYSICAL_REPEATS == 3
    assert xr1.BOOTSTRAP_SEED == 20_260_903


def test_parser_exposes_only_the_declared_stages():
    parser = xr1.build_parser()
    args = parser.parse_args([])
    assert args.stage == "all"
    for stage in ("preflight", "physical", "bootstrap", "top20", "all"):
        assert parser.parse_args(["--stage", stage]).stage == stage
    with pytest.raises(SystemExit):
        parser.parse_args(["--stage", "not-a-stage"])
