from __future__ import annotations

import itertools

import numpy as np
import pytest

from experiments import ddm_gf3_gop_histogram_globality_audit as audit


def test_candidate_pairings_are_disjoint_and_within_gop() -> None:
    center = np.arange(40, dtype=np.int64).reshape(8, 5)
    rows = audit.candidate_pairings([1, 2, 3, 4, 5], center)
    assert set(rows) == {"fixed_extremes", *(f"extreme_center_class_{label}" for label in range(5))}
    for pairs in rows.values():
        flat = [frame for pair in pairs for frame in pair]
        assert len(pairs) == 2
        assert len(flat) == len(set(flat))
        assert set(flat).issubset({1, 2, 3, 4, 5})


def test_histogram_tv_pair_bound_never_exceeds_exact_shared_histogram_error() -> None:
    histograms = np.array(
        [
            [[4, 0, 0, 0, 0], [2, 2, 0, 0, 0]],
            [[0, 4, 0, 0, 0], [1, 3, 0, 0, 0]],
        ],
        dtype=np.int32,
    )
    bound = audit.exact_pairing_row(histograms, [[0, 1]])["lower_bound"]
    exact = min(
        sum(max(int(left[label]), int(right[label])) for label in range(5)) - int(left.sum())
        for left, right in itertools.product(histograms[0], histograms[1])
    )
    assert bound == exact


def test_exact_gop_certificate_selects_strongest_valid_pairing(monkeypatch) -> None:
    monkeypatch.setattr(audit.gf3, "N_PAIRS", 4)
    monkeypatch.setattr(audit.gf3, "NUM_CLASSES", 5)
    translations = np.array([[0, 0]], dtype=np.int16)
    histograms = np.array(
        [
            [[10, 0, 0, 0, 0]],
            [[9, 1, 0, 0, 0]],
            [[0, 10, 0, 0, 0]],
            [[1, 9, 0, 0, 0]],
        ],
        dtype=np.int32,
    )
    certificate = audit.exact_gop_certificate(histograms, translations, gop_length=4)
    gop = certificate["gops"][0]
    candidate_bounds = [row["lower_bound"] for row in gop["candidate_pairings"].values()]
    assert gop["selected_lower_bound"] == max(candidate_bounds)
    assert certificate["certified_lower_bound_mismatches"] == max(candidate_bounds)


def test_exact_gop_certificate_refuses_nondividing_length(monkeypatch) -> None:
    monkeypatch.setattr(audit.gf3, "N_PAIRS", 4)
    histograms = np.zeros((4, 1, 5), dtype=np.int32)
    with pytest.raises(audit.GF3GlobalityError, match="divide"):
        audit.exact_gop_certificate(histograms, np.array([[0, 0]], dtype=np.int16), gop_length=3)


def test_price_row_refuses_when_bound_repair_charge_exceeds_cap() -> None:
    certificate = {"gop_length": 10, "certified_lower_bound_mismatches": 270_034}
    physical = {
        "pricing": {"packet_bytes": 31_896, "residual_bytes": 355_124},
        "observed_fit_upper_bound": {"post_sweep_mismatches": 1_281_107},
    }
    row = audit.price_row(certificate, physical)
    assert row["charter_optimistic_repair_charge"]["packet_plus_charge_bytes"] == 96_833.607
    assert row["disposition"] == "PRICING-REFUSED"


def test_price_row_admits_only_as_bound_when_both_direct_gates_pass() -> None:
    certificate = {"gop_length": 10, "certified_lower_bound_mismatches": 40_000}
    physical = {
        "pricing": {"packet_bytes": 60_000, "residual_bytes": 90_000},
        "observed_fit_upper_bound": {"post_sweep_mismatches": 1_000_000},
    }
    row = audit.price_row(certificate, physical)
    assert row["direct_bound_gate_passes"]
    assert row["disposition"] == "BOUND-ADMITS-BUILD"
