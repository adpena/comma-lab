"""Unit tests for RXC1 orchestration; physical identity is the retained n600 gate."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from tac.payload_retention_gate import check_no_measure_and_discard_payload

REPO = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO / "experiments/ddm_rxc1_restartable_exact_coder.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "ddm_rxc1_restartable_exact_coder", MODULE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["ddm_rxc1_restartable_exact_coder"] = module
    spec.loader.exec_module(module)
    return module


rxc1 = _load_module()


def test_preregistered_sample_has_32_unique_pairs():
    rows = rxc1.sample_pairs()
    assert len(rows) == 32
    assert len({row["pair"] for row in rows}) == 32


def test_preregistered_sample_has_four_pairs_per_stratum():
    rows = rxc1.sample_pairs()
    assert [sum(row["stratum"] == value for row in rows) for value in range(8)] == [4] * 8


def test_preregistered_sample_is_seed_deterministic():
    assert rxc1.sample_pairs() == rxc1.sample_pairs(rxc1.SCREEN_SEED)
    assert rxc1.sample_pairs() != rxc1.sample_pairs(rxc1.SCREEN_SEED + 1)


def test_rankdata_uses_average_ranks_for_ties():
    values = np.array([4.0, 1.0, 4.0, 2.0])
    assert np.array_equal(rxc1.rankdata(values), [3.5, 1.0, 3.5, 2.0])


def test_correlation_is_one_for_an_affine_copy():
    values = np.arange(12, dtype=np.float64)
    assert rxc1.correlation(values, 3.0 * values + 7.0) == pytest.approx(1.0)


def test_correlation_refuses_a_constant_column():
    with pytest.raises(rxc1.Rxc1Error, match="undefined"):
        rxc1.correlation(np.ones(8), np.arange(8))


def test_nearest_checkpoint_is_at_or_before_pair():
    assert rxc1.RestartableExactCoder.nearest_checkpoint(0, 200) == 0
    assert rxc1.RestartableExactCoder.nearest_checkpoint(199, 200) == 0
    assert rxc1.RestartableExactCoder.nearest_checkpoint(200, 200) == 200
    assert rxc1.RestartableExactCoder.nearest_checkpoint(599, 200) == 400


def test_nearest_checkpoint_refuses_invalid_pair_and_stride():
    with pytest.raises(rxc1.Rxc1Error, match="pair"):
        rxc1.RestartableExactCoder.nearest_checkpoint(600, 200)
    with pytest.raises(rxc1.Rxc1Error, match="stride"):
        rxc1.RestartableExactCoder.nearest_checkpoint(3, 0)


def test_checkpoint_frames_cover_both_stride_lattices():
    for stride in rxc1.STRIDES:
        assert set(range(0, rxc1.PAIR_COUNT, stride)).issubset(rxc1.CHECKPOINT_FRAMES)
    assert rxc1.PAIR_COUNT in rxc1.CHECKPOINT_FRAMES


def test_compare_bytes_counts_length_difference(tmp_path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.write_bytes(b"abc")
    right.write_bytes(b"abd!")
    comparison = rxc1.compare_bytes(left, right)
    assert comparison["compared_bytes"] == 4
    assert comparison["differing_bytes"] == 2
    assert comparison["byte_identical"] is False


def test_create_edit_payloads_persists_one_changed_token_per_pair(tmp_path):
    class Tokens:
        def __getitem__(self, _pair):
            return np.zeros((384, 512), dtype=np.uint8)

    rows = rxc1.create_edit_payloads(Tokens(), tmp_path)
    assert len(rows) == rxc1.SCREEN_N
    for row in rows:
        assert row["changed_tokens"] == 1
        path = Path(row["payload"]["path"])
        with np.load(path, allow_pickle=False) as blob:
            plane = blob[str(row["pair"])]
        assert np.count_nonzero(plane) == 1
        assert row["old_token"] == 0
        assert row["new_token"] in {1, 2, 3, 4}


def test_create_edit_payloads_reuses_only_byte_identical_preregistration(tmp_path):
    class Tokens:
        def __getitem__(self, _pair):
            return np.zeros((384, 512), dtype=np.uint8)

    first = rxc1.create_edit_payloads(Tokens(), tmp_path)
    second = rxc1.create_edit_payloads(Tokens(), tmp_path)
    assert [row["payload"]["sha256"] for row in first] == [
        row["payload"]["sha256"] for row in second
    ]


def test_stride_stats_reports_exact_identity():
    rows = []
    for value in (-2, -1, 0, 1, 2):
        rows.append(
            {
                "full_delta_bytes": value,
                "incremental": {
                    "200": {
                        "archive_delta_bytes": value,
                        "wall_seconds": 10.0,
                        "frames_encoded": 300,
                    }
                },
                "stream_comparisons": {"200": {"byte_identical": True}},
            }
        )
    stats = rxc1._stride_stats(rows, 200)
    assert stats["pearson"] == pytest.approx(1.0)
    assert stats["spearman"] == pytest.approx(1.0)
    assert stats["max_abs_error_bytes"] == 0.0
    assert stats["sign_agreement_count"] == 5
    assert stats["stream_identity_count"] == 5
    assert stats["gate_pass"] is True


def test_parser_requires_an_explicit_stage():
    with pytest.raises(SystemExit):
        rxc1.build_parser().parse_args([])


def test_exact_delta_refuses_a_pair_that_does_not_match_the_payload(tmp_path):
    edit = tmp_path / "edit.npz"
    np.savez(edit, **{"7": np.zeros((384, 512), dtype=np.uint8)})
    api = object.__new__(rxc1.RestartableExactCoder)
    with pytest.raises(rxc1.Rxc1Error, match="does not match"):
        api.exact_delta(edit, 8, 200, tmp_path / "run")


def test_exact_coder_sources_pass_payload_retention_gate():
    findings = check_no_measure_and_discard_payload(
        repo_root=REPO,
        strict=False,
        roots=(
            "experiments/ddm_jg2_tail_reencode.py",
            "experiments/ddm_rxc1_restartable_exact_coder.py",
        ),
    )
    assert findings == []
