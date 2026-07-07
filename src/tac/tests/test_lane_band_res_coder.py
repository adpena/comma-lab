# SPDX-License-Identifier: MIT
"""Tests for the LBND4 lane-coeff section codec (Mallat/Ballé review row 7 BUILD 1).

LBND4 = the LBND2 quantization grid (identical dequantized statistic, asserted) + the ξ
delta/context residual entropy stage (``tac.boundary_math.xi_spline_residual_coder``,
best-of-three {varint, zlib9, rice}). Locks: bit-identity roundtrip per scheme, the
NO-FAKE encode self-check, decode-reencode byte-identity, format-agnostic dispatch,
grid-preserving re-serialize, fail-closed parsing, and the gauge DSL leg
(``LaneBandCoderGauge`` + ``lane_band_coder_byte_close_flags``). Synthetic LaneLines are
used for hermetic CORRECTNESS only; the n600 MEASURED evidence lives at
``experiments/results/lane_band_res_coder_20260707/lane_band_res_coder_n600_measured.json``.
"""
from __future__ import annotations

import json
import struct

import numpy as np
import pytest

from tac.boundary_math.analytic_lane_render_band import (
    LANE_BAND_RES_MAGIC,
    LaneBandRenderConfig,
    deserialize_lane_band_any,
    deserialize_lane_band_rd,
    deserialize_lane_band_res,
    render_config_from_header,
    serialize_lane_band_any,
    serialize_lane_band_rd,
    serialize_lane_band_res,
)
from tac.boundary_math.lane_sdf_component import LaneLine
from tac.boundary_math.xi_spline_residual_coder import (
    RESIDUAL_SCHEMES,
    XiSplineResidualError,
    decode_residual_matrix,
    encode_residual_matrix,
    residual_scheme_id,
    residual_scheme_name,
)


def _line(c0: float, *, dash: bool, fr=(5.0, 60.0)) -> LaneLine:
    return LaneLine(
        centerline_coeffs=np.array([1e-5, -2e-4, 0.01, c0], np.float64),
        halfwidth_coeffs=np.array([0.001, 1.5], np.float64),
        dash_period_m=(6.0 if dash else 0.0),
        dash_phase_m=(1.25 if dash else 0.0),
        dash_duty=(0.5 if dash else 0.5),
        forward_range=(float(fr[0]), float(fr[1])),
    )


def _pairs(n: int = 12) -> list[list[LaneLine]]:
    """Slowly drifting 2-line manifold + one absent-slot pair (exercises carry-forward hold)."""
    rng = np.random.default_rng(0)
    out: list[list[LaneLine]] = []
    for p in range(n):
        drift = 0.02 * p + float(rng.normal(0.0, 0.005))
        lines = [_line(-1.8 + drift, dash=True), _line(1.8 - drift, dash=False)]
        if p == n // 2:
            lines = lines[:1]  # absent slot -> presence hold path
        out.append(lines)
    return out


_CFG = LaneBandRenderConfig()


def _slot_vecs(pairs_lines):
    from tac.boundary_math.analytic_lane_render_band import _line_to_slot_vec

    return [[_line_to_slot_vec(ln) for ln in lines] for lines in pairs_lines]


def _assert_lines_equal(a, b):
    va, vb = _slot_vecs(a), _slot_vecs(b)
    assert len(va) == len(vb)
    for la, lb in zip(va, vb):
        assert len(la) == len(lb)
        for x, y in zip(la, lb):
            assert np.array_equal(x, y)


# --------------------------------------------------------------------------- #
# roundtrip + statistic identity
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("scheme", list(RESIDUAL_SCHEMES))
def test_roundtrip_bit_identity_per_scheme(scheme):
    pl = _pairs()
    blob = serialize_lane_band_res(pl, _CFG, scheme=scheme)
    dec, hdr = deserialize_lane_band_res(blob)
    assert hdr["res"]["scheme"] == scheme
    # the decoded statistic equals the LBND2 dequantized statistic (same grid)
    rd_dec, _ = deserialize_lane_band_rd(serialize_lane_band_rd(pl, _CFG))
    _assert_lines_equal(dec, rd_dec)


def test_auto_scheme_pick_is_deterministic_and_recorded():
    pl = _pairs()
    b1 = serialize_lane_band_res(pl, _CFG)
    b2 = serialize_lane_band_res(pl, _CFG)
    assert b1 == b2
    _, hdr = deserialize_lane_band_res(b1)
    assert hdr["res"]["scheme"] in RESIDUAL_SCHEMES
    assert hdr["format"] == 4
    assert "rd" in hdr  # LBND2-compatible grid block (grid-preserving re-serialize)


def test_decode_reencode_byte_identical():
    pl = _pairs()
    blob = serialize_lane_band_res(pl, _CFG)
    dec, hdr = deserialize_lane_band_res(blob)
    re_blob = serialize_lane_band_any(dec, render_config_from_header(hdr), hdr)
    assert re_blob == blob


def test_header_render_config_roundtrip():
    pl = _pairs()
    cfg = LaneBandRenderConfig(softness=0.7, weight=0.9, u_mask_enabled=True,
                               u_mask_tau=0.8, u_mask_eps=0.3)
    _, hdr = deserialize_lane_band_res(serialize_lane_band_res(pl, cfg))
    rc = render_config_from_header(hdr)
    assert rc.softness == 0.7 and rc.weight == 0.9 and rc.u_mask_enabled
    assert rc.u_mask_tau == 0.8 and rc.u_mask_eps == 0.3


def test_any_dispatch_recognizes_lbnd4():
    pl = _pairs()
    blob = serialize_lane_band_res(pl, _CFG)
    dec_any, hdr_any = deserialize_lane_band_any(blob)
    dec_res, hdr_res = deserialize_lane_band_res(blob)
    assert hdr_any == hdr_res
    _assert_lines_equal(dec_any, dec_res)


def test_empty_pairs_roundtrip():
    blob = serialize_lane_band_res([[], [], []], _CFG)
    dec, hdr = deserialize_lane_band_res(blob)
    assert dec == [[], [], []]
    assert int(hdr["rd"]["n_pairs"]) == 3 and int(hdr["rd"]["K"]) == 0


def test_grid_preserved_on_subset_reserialize():
    """A capped subset re-serialized via serialize_lane_band_any lands on the SAME grid +
    scheme (the capped-inflate gate contract)."""
    pl = _pairs(10)
    blob = serialize_lane_band_res(pl, _CFG)
    dec, hdr = deserialize_lane_band_res(blob)
    cap = serialize_lane_band_any(dec[:4], render_config_from_header(hdr), hdr)
    dec_cap, hdr_cap = deserialize_lane_band_res(cap)
    assert hdr_cap["res"]["scheme"] == hdr["res"]["scheme"]
    assert hdr_cap["rd"]["base_steps"] == hdr["rd"]["base_steps"]
    _assert_lines_equal(dec_cap, dec[:4])


# --------------------------------------------------------------------------- #
# fail-closed parsing
# --------------------------------------------------------------------------- #
def test_bad_magic_refused():
    with pytest.raises(ValueError, match="LBND4 magic"):
        deserialize_lane_band_res(b"NOTLB\x00" + b"\x00" * 32)


def test_trailing_bytes_refused():
    blob = serialize_lane_band_res(_pairs(4), _CFG) + b"\x00"
    with pytest.raises(ValueError, match="trailing"):
        deserialize_lane_band_res(blob)


def test_unknown_scheme_refused_at_encode():
    with pytest.raises(ValueError, match="unknown LBND4 residual scheme"):
        serialize_lane_band_res(_pairs(4), _CFG, scheme="arith")


def test_header_is_parseable_json_with_res_block():
    blob = serialize_lane_band_res(_pairs(4), _CFG)
    (hlen,) = struct.unpack_from("<I", blob, len(LANE_BAND_RES_MAGIC))
    hdr = json.loads(blob[len(LANE_BAND_RES_MAGIC) + 4:
                          len(LANE_BAND_RES_MAGIC) + 4 + hlen].decode("utf-8"))
    assert hdr["res"]["scheme"] in RESIDUAL_SCHEMES


# --------------------------------------------------------------------------- #
# the shared ξ residual entropy stage (public wrappers — single-source discipline)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("scheme", list(RESIDUAL_SCHEMES))
def test_public_residual_matrix_roundtrip(scheme):
    rng = np.random.default_rng(1)
    res = rng.integers(-300, 300, size=(37, 22)).astype(np.int64)
    blob = encode_residual_matrix(res, scheme)
    back = decode_residual_matrix(blob, residual_scheme_id(scheme), 37, 22)
    assert np.array_equal(back, res)
    assert residual_scheme_name(residual_scheme_id(scheme)) == scheme


def test_scheme_id_name_fail_closed():
    with pytest.raises(XiSplineResidualError):
        residual_scheme_id("arith")
    with pytest.raises(XiSplineResidualError):
        residual_scheme_name(99)


# --------------------------------------------------------------------------- #
# gauge DSL leg (LaneBandCoderGauge — the byte-close-side chart, never trainer argv)
# --------------------------------------------------------------------------- #
def test_gauge_chart_and_accessor():
    from tac.witness_dsl.gauge import (
        COMPONENT_GAUGES,
        GaugeComponent,
        LaneBandCoderGauge,
        component_of,
        lane_band_coder_byte_close_flags,
    )

    assert COMPONENT_GAUGES[GaugeComponent.LANE_BAND_CODER] is LaneBandCoderGauge
    assert component_of(LaneBandCoderGauge.RES) is GaugeComponent.LANE_BAND_CODER
    assert lane_band_coder_byte_close_flags(LaneBandCoderGauge.RD) == ()
    assert lane_band_coder_byte_close_flags(LaneBandCoderGauge.RES) == ("--lane-band-res",)
    assert lane_band_coder_byte_close_flags(LaneBandCoderGauge.NAIVE) == ("--lane-band-naive",)


def test_byte_close_flags_stay_out_of_trainer_surface():
    """never-invent-flags: the byte-close-tool flags must NOT be DSL-EMITTED trainer flags
    (the lever_registry ``stale == []`` invariant)."""
    from tac.witness_dsl.lever_registry import dsl_emitted_flags

    emitted = dsl_emitted_flags()
    assert "--lane-band-res" not in emitted
    assert "--lane-band-naive" not in emitted


def test_byte_close_tool_actually_exposes_the_flags():
    """Grep the REAL byte-close tool argparse for the chart flags (never-invent-flags on the
    TOOL surface — the same discipline the trainer parse-test applies to trainer flags)."""
    from pathlib import Path

    tool = Path(__file__).resolve().parents[3] / "tools" / "levelset_byte_close_and_eval.py"
    src = tool.read_text()
    assert '"--lane-band-res"' in src
    assert '"--lane-band-naive"' in src
