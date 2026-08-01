"""Tests for ddm_rh1 token-field rate decomposition and the SMEVR base-rule race.

These verify BEHAVIOUR, not constants: the central invariant is that
``framed_bytes`` under the deterministic mode reproduces ``encode_token_codes``
byte-for-byte, so every alternative base is scored on the same ruler as the
shipped member.
"""

from __future__ import annotations

import io
import json
import sys
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_REPO_ROOT / "src"), str(_REPO_ROOT / "experiments")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ddm_r7_token_coder as r7
from ddm_rh1_token_field_rate_decomposition import (
    RATE_DENOMINATOR,
    RH1Error,
    decompose_field_gap,
    framed_bytes,
    load_token_field,
    propose_base,
    race_base_rule,
    rate_delta_s,
)

LEVELS = 4
SHAPE = (12, 3, 4, 2)


def _field(seed: int, *, sparsity: float = 0.7) -> np.ndarray:
    """A temporally sparse lattice: a shared base plus occasional events."""
    rng = np.random.default_rng(seed)
    base = rng.integers(0, LEVELS, size=SHAPE[1:], dtype=np.uint8)
    out = np.broadcast_to(base, SHAPE).copy()
    events = rng.random(SHAPE) > sparsity
    out[events] = rng.integers(0, LEVELS, size=int(events.sum()), dtype=np.uint8)
    return np.ascontiguousarray(out)


# ------------------------------------------------------- the ruler must be shared


def test_mode_base_framed_bytes_reproduces_encode_token_codes_exactly() -> None:
    values = _field(0)
    base, _ = r7.factor_mode_delta(values, LEVELS)
    framed, base_stream, delta_stream = framed_bytes(values, base, LEVELS)
    shipped = r7.encode_token_codes(values, levels=LEVELS, codec="smevr")
    assert framed == len(shipped)
    assert r7.HEADER.size + base_stream + delta_stream == len(shipped)


def test_every_proposed_base_is_lossless_and_priced_on_the_same_ruler() -> None:
    values = _field(1)
    for alpha, exponent in ((0.0, 1.0), (0.0, 2.0), (2.0, 0.5)):
        base = propose_base(values, alpha=alpha, exponent=exponent, levels=LEVELS)
        delta = (values.astype(np.int16) - base[None].astype(np.int16)) % LEVELS
        restored = r7.reconstruct_mode_delta(base, delta.astype(np.uint8), LEVELS)
        assert np.array_equal(restored, values)
        assert framed_bytes(values, base, LEVELS)[0] > r7.HEADER.size


def test_framed_bytes_rejects_a_base_that_does_not_reconstruct() -> None:
    values = _field(2)
    base, _ = r7.factor_mode_delta(values, LEVELS)
    bad = base.astype(np.uint8) + np.uint8(LEVELS)  # out of the declared lattice
    with pytest.raises((RH1Error, r7.DDMR7CoderError, ValueError)):
        framed_bytes(values, bad, LEVELS)


# ------------------------------------------------------------------ decomposition


def test_gap_of_a_field_with_itself_is_zero_on_every_leg() -> None:
    values = _field(3)
    gap = decompose_field_gap(values, values, levels=LEVELS)
    assert gap.total_byte_delta == 0
    assert gap.event_swap_byte_delta == 0
    assert gap.base_swap_byte_delta == 0
    assert gap.interaction_byte_delta == 0
    assert gap.left_event_rate == pytest.approx(gap.right_event_rate)


def test_decomposition_legs_sum_to_the_total_by_construction() -> None:
    gap = decompose_field_gap(_field(4), _field(5, sparsity=0.9), levels=LEVELS)
    assert (
        gap.event_swap_byte_delta + gap.base_swap_byte_delta + gap.interaction_byte_delta
        == gap.total_byte_delta
    )


def test_a_sparser_field_codes_smaller_and_reports_a_lower_event_rate() -> None:
    dense, sparse = _field(6, sparsity=0.4), _field(6, sparsity=0.97)
    gap = decompose_field_gap(dense, sparse, levels=LEVELS)
    assert gap.right_event_rate < gap.left_event_rate
    assert gap.total_byte_delta < 0


def test_mismatched_geometry_is_refused_rather_than_silently_compared() -> None:
    other = np.zeros((11, 3, 4, 2), dtype=np.uint8)
    with pytest.raises(RH1Error, match="geometry differs"):
        decompose_field_gap(_field(7), other, levels=LEVELS)


# --------------------------------------------------------------------- base race


def test_large_alpha_recovers_the_incumbent_mode() -> None:
    values = _field(8)
    mode, _ = r7.factor_mode_delta(values, LEVELS)
    assert np.array_equal(propose_base(values, alpha=1e9, exponent=1.0, levels=LEVELS), mode)


def test_propose_base_refuses_values_outside_the_declared_lattice() -> None:
    """An out-of-lattice value would otherwise vanish from every bin, silently."""
    values = _field(8).copy()
    values[0, 0, 0, 0] = np.uint8(LEVELS + 1)
    with pytest.raises(RH1Error, match="leave the declared lattice"):
        propose_base(values, alpha=0.0, exponent=1.0, levels=LEVELS)


def test_propose_base_raises_no_runtime_warning() -> None:
    """Pins the BLAS-free reduction: the matmul form emitted spurious FPU warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        propose_base(_field(8), alpha=0.0, exponent=2.0, levels=LEVELS)


def test_propose_base_matches_a_brute_force_per_cell_argmin() -> None:
    """The vectorised reduction must equal a literal per-cell search, not merely run."""
    values, exponent = _field(9), 2.0
    circ = np.array([min(r, LEVELS - r) for r in range(LEVELS)], dtype=np.float64)
    cost = np.where(np.arange(LEVELS) == 0, 0.0, circ**exponent)
    got = propose_base(values, alpha=0.0, exponent=exponent, levels=LEVELS).reshape(-1)
    flat = values.reshape(values.shape[0], -1)
    for cell in range(flat.shape[1]):
        totals = [sum(cost[(int(s) - b) % LEVELS] for s in flat[:, cell]) for b in range(LEVELS)]
        assert int(got[cell]) == int(np.argmin(totals))


def test_race_includes_the_incumbent_at_zero_delta_and_sorts_by_bytes() -> None:
    rows = race_base_rule(_field(9), levels=LEVELS, alphas=(0.0, 4.0), exponents=(1.0, 2.0))
    incumbent = [row for row in rows if row["rule"] == "mode (incumbent)"]
    assert len(incumbent) == 1
    assert incumbent[0]["byte_delta"] == 0
    assert incumbent[0]["cells_moved"] == 0
    assert [int(row["framed"]) for row in rows] == sorted(int(row["framed"]) for row in rows)


def test_race_byte_delta_is_consistent_with_the_incumbent_framed_size() -> None:
    values = _field(10)
    rows = race_base_rule(values, levels=LEVELS, alphas=(0.0,), exponents=(2.0,))
    incumbent = next(row for row in rows if row["rule"] == "mode (incumbent)")
    for row in rows:
        assert int(row["framed"]) - int(incumbent["framed"]) == int(row["byte_delta"])
        assert float(row["rate_delta_s"]) == pytest.approx(rate_delta_s(int(row["byte_delta"])))


def test_race_rows_are_strict_json_serialisable() -> None:
    """A receipt must survive a strict parser: `float('inf')` emits a bare `Infinity`."""
    rows = race_base_rule(_field(12), levels=LEVELS, alphas=(0.0, 2.0), exponents=(1.0,))
    raw = json.dumps(rows)
    assert "Infinity" not in raw and "NaN" not in raw

    def _reject(constant: str) -> object:
        raise AssertionError(f"non-strict JSON constant emitted: {constant}")

    assert json.loads(raw, parse_constant=_reject) == rows


def test_rate_delta_s_uses_the_contest_denominator() -> None:
    assert rate_delta_s(RATE_DENOMINATOR) == pytest.approx(25.0)
    assert rate_delta_s(0) == 0.0
    assert rate_delta_s(-1000) < 0.0


# ---------------------------------------------------------------------------- io


def test_load_token_field_round_trips_a_dr7t_member(tmp_path: Path) -> None:
    values = _field(11)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr("state/tokens.dr7t", r7.encode_token_codes(values, levels=LEVELS, codec="smevr"))
    path = tmp_path / "archive.zip"
    path.write_bytes(buf.getvalue())
    assert np.array_equal(load_token_field(path), values)
