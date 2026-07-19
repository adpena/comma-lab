# SPDX-License-Identifier: MIT
"""Tests for the #425 dash-phase carrier (curve-domain per-dash δ(s) codec)."""
from __future__ import annotations

import numpy as np
import pytest

from tac.boundary_math.dash_phase_carrier import (
    DASH_PHASE_MAGIC,
    DashPhaseConfig,
    DashPhaseError,
    JITTER_PRIOR_SYMBOL_PROBS,
    _BitReader,
    _BitWriter,
    _canonical_codes,
    build_prior_huffman_lengths,
    decode_dash_phase_carrier,
    encode_dash_phase_carrier,
    expected_bits_per_symbol,
    extract_dash_observations,
)

H, W = 96, 128
LANE = 1
ROAD = 0


def _blank() -> np.ndarray:
    return np.full((H, W), ROAD, dtype=np.int64)


def _paint_dash(lstar: np.ndarray, r: int, c: int, h: int = 3, w: int = 8) -> None:
    lstar[r : r + h, c : c + w] = LANE


def _zero_xi(p: int) -> np.ndarray:
    return np.zeros((p, 6), dtype=np.float64)


CFG = DashPhaseConfig(match_radius_px=6.0)


# --------------------------------------------------------------------------- #
# prior-derived code
# --------------------------------------------------------------------------- #
def test_prior_probs_sum_to_one():
    assert abs(sum(JITTER_PRIOR_SYMBOL_PROBS.values()) - 1.0) < 1e-12


def test_huffman_lengths_kraft_and_expected_bits():
    lengths = build_prior_huffman_lengths()
    # Kraft equality for a complete Huffman code
    assert abs(sum(2.0 ** -ln for ln in lengths.values()) - 1.0) < 1e-12
    # zero-offset symbol is the most probable -> shortest code
    assert lengths["0"] == min(lengths.values())
    # pre-registered expectation: 2.267 bits/component (stated in the module docstring)
    assert abs(expected_bits_per_symbol(lengths) - 2.267) < 5e-3


def test_canonical_codes_prefix_free():
    codes = _canonical_codes(build_prior_huffman_lengths())
    vals = [(format(c, f"0{n}b")) for c, n in codes.values()]
    for i, a in enumerate(vals):
        for j, b in enumerate(vals):
            if i != j:
                assert not b.startswith(a)


def test_bitio_roundtrip_varint_and_bits():
    bw = _BitWriter()
    bw.write(0b1011, 4)
    for v in (0, 1, 127, 128, 300, 2**20):
        bw.write_varint(v)
    blob = bw.getvalue()
    br = _BitReader(blob)
    assert br.read(4) == 0b1011
    for v in (0, 1, 127, 128, 300, 2**20):
        assert br.read_varint() == v
    br.assert_fully_consumed()


def test_bitreader_refuses_unconsumed_bytes():
    bw = _BitWriter()
    bw.write(0xABCD, 16)
    br = _BitReader(bw.getvalue())
    br.read(4)
    with pytest.raises(DashPhaseError, match="unconsumed"):
        br.assert_fully_consumed()


# --------------------------------------------------------------------------- #
# extraction
# --------------------------------------------------------------------------- #
def test_extract_dash_observations_basic():
    lstar = _blank()
    _paint_dash(lstar, 30, 40)
    _paint_dash(lstar, 60, 80)
    obs = extract_dash_observations(lstar, CFG)
    assert len(obs) == 2
    r0, c0 = obs[0].centroid_rc
    assert abs(r0 - 31.0) < 1e-9 and abs(c0 - 43.5) < 1e-9
    assert obs[0].area == 24
    # horizontal dash -> tilt near 0 (mod pi)
    assert min(obs[0].tilt, np.pi - obs[0].tilt) < 0.2


def test_extract_filters_border_and_small():
    lstar = _blank()
    _paint_dash(lstar, 1, 40)          # touches border band -> excluded
    lstar[50, 50] = LANE               # area 1 < min_area -> excluded
    _paint_dash(lstar, 40, 40)         # kept
    obs = extract_dash_observations(lstar, CFG)
    assert len(obs) == 1


# --------------------------------------------------------------------------- #
# codec end-to-end
# --------------------------------------------------------------------------- #
def _frames_translating(n: int, dc_per_frame: int = 1) -> np.ndarray:
    out = np.stack([_blank() for _ in range(n)])
    for p in range(n):
        _paint_dash(out[p], 40, 40 + dc_per_frame * p)
        _paint_dash(out[p], 60, 70)
    return out


def test_encode_decode_bit_identical_and_magic():
    ls = _frames_translating(5)
    section, report, dec = encode_dash_phase_carrier(ls, _zero_xi(5), CFG)
    assert section[:6] == DASH_PHASE_MAGIC
    assert report.reconstruction_bit_identical
    # decode again independently
    dec2 = decode_dash_phase_carrier(section)
    assert len(dec2) == 5
    for fa, fb in zip(dec, dec2):
        assert [(d.track_id, d.centroid_rc, d.born) for d in fa] == [
            (d.track_id, d.centroid_rc, d.born) for d in fb
        ]


def test_tracking_matched_deltas_recover_translation():
    ls = _frames_translating(6, dc_per_frame=1)
    _, report, dec = encode_dash_phase_carrier(ls, _zero_xi(6), CFG)
    # 2 dashes, 6 frames: frame0 = 2 births; frames 1..5 = 2 matches each (radius 6 > 1px/frame)
    assert report.n_births == 2
    assert report.n_matched == 10
    assert report.n_deaths == 0 and report.n_rebirths == 0
    # decoded centroid of the moving dash tracks the observed translation to <= q/sqrt(2)
    for p in range(6):
        moving = [d for d in dec[p] if d.track_id == 0][0]
        assert abs(moving.centroid_rc[1] - (43.5 + p)) <= 1.0
        assert abs(moving.centroid_rc[0] - 41.0) <= 1.0


def test_death_and_rebirth_uses_dormant_pool():
    frames = []
    for p in range(6):
        f = _blank()
        _paint_dash(f, 40, 40)          # persistent dash
        if p not in (2, 3):             # blinking dash: dies at p=2,3, back at p=4
            _paint_dash(f, 60, 70)
        frames.append(f)
    ls = np.stack(frames)
    _, report, dec = encode_dash_phase_carrier(ls, _zero_xi(6), CFG)
    assert report.n_deaths == 1
    assert report.n_rebirths == 1          # the blink-back is a REBIRTH, not a new anchor
    assert report.n_births == 2            # only the two original anchors ever paid full price
    assert report.blink_back_fraction == 1.0
    # the reborn dash keeps its ORIGINAL track id (world-frame identity)
    tids_p4 = {d.track_id for d in dec[4]}
    tids_p0 = {d.track_id for d in dec[0]}
    assert tids_p4 == tids_p0


def test_dormant_expiry_forces_new_anchor():
    cfg = DashPhaseConfig(match_radius_px=6.0, dormant_max_frames=1)
    frames = []
    for p in range(7):
        f = _blank()
        _paint_dash(f, 40, 40)
        if p not in (2, 3, 4):          # gone 3 frames > dormant_max 1
            _paint_dash(f, 60, 70)
        frames.append(f)
    _, report, _ = encode_dash_phase_carrier(np.stack(frames), _zero_xi(7), cfg)
    assert report.n_rebirths == 0
    assert report.n_births == 3            # expired dormant -> full birth again


def test_esc_path_large_jump_within_radius():
    # a 5px jump: within match radius 6 -> matched, |delta| > 2 -> ESC symbol
    ls = _frames_translating(3, dc_per_frame=5)
    _, report, _ = encode_dash_phase_carrier(ls, _zero_xi(3), CFG)
    assert report.symbol_histogram["ESC"] >= 2
    assert report.esc_rate > 0.0


def test_include_xi_false_needs_external_and_matches():
    ls = _frames_translating(4)
    xi = np.zeros((4, 6))
    cfg_no_xi = DashPhaseConfig(match_radius_px=6.0, include_xi=False)
    sec_no, rep_no, dec_no = encode_dash_phase_carrier(ls, xi, cfg_no_xi)
    sec_yes, rep_yes, _ = encode_dash_phase_carrier(ls, xi, CFG)
    assert rep_no.xi_bytes == 0 and rep_yes.xi_bytes == 4 * 6 * 2
    assert len(sec_no) < len(sec_yes)  # the composed form drops the already-banked dxi bytes
    with pytest.raises(DashPhaseError, match="xi_twists_external"):
        decode_dash_phase_carrier(sec_no)
    dec2 = decode_dash_phase_carrier(
        sec_no, xi_twists_external=xi.astype(np.float16).astype(np.float64)
    )
    assert [(d.track_id, d.centroid_rc) for f in dec2 for d in f] == [
        (d.track_id, d.centroid_rc) for f in dec_no for d in f
    ]


def test_determinism_same_input_same_bytes():
    ls = _frames_translating(5)
    s1, _, _ = encode_dash_phase_carrier(ls, _zero_xi(5), CFG)
    s2, _, _ = encode_dash_phase_carrier(ls, _zero_xi(5), CFG)
    assert s1 == s2


def test_corrupt_magic_and_truncation_refuse():
    ls = _frames_translating(3)
    section, _, _ = encode_dash_phase_carrier(ls, _zero_xi(3), CFG)
    with pytest.raises(DashPhaseError, match="magic"):
        decode_dash_phase_carrier(b"XXXX" + section[4:])
    with pytest.raises(DashPhaseError):
        decode_dash_phase_carrier(section + b"\x00\x00")  # trailing seed bytes refused


def test_empty_lane_frames_ok():
    ls = np.stack([_blank() for _ in range(3)])
    section, report, dec = encode_dash_phase_carrier(ls, _zero_xi(3), CFG)
    assert report.n_births == 0 and report.n_matched == 0
    assert all(len(f) == 0 for f in dec)
    assert decode_dash_phase_carrier(section) == [[], [], []]


def test_report_bit_accounting_consistent():
    ls = _frames_translating(6)
    _, r, _ = encode_dash_phase_carrier(ls, _zero_xi(6), CFG)
    total_bits = r.alive_bits + r.delta_bits + r.birth_bits + r.rebirth_bits
    # stream also carries per-frame n_new varints (8 bits each here) -> stream_bytes*8 >= total
    assert r.stream_bytes * 8 >= total_bits
    assert r.stream_bytes * 8 - total_bits < 8 * (r.n_frames + 2) + 8
    # prior-derived code must not exceed the zlib9 stage on this tiny near-zero-delta stream
    assert r.prior_code_delta_bytes <= max(1, r.zlib9_delta_stream_bytes)


def test_xi_advection_used_by_prediction():
    # forward translation twist: ground homography shifts points; encode must stay closed-loop
    ls = _frames_translating(4, dc_per_frame=0)
    xi = np.zeros((4, 6))
    xi[:, 0] = 0.5  # forward motion rho_x
    section, report, _ = encode_dash_phase_carrier(ls, xi, CFG)
    assert report.reconstruction_bit_identical
    dec = decode_dash_phase_carrier(section)
    assert len(dec) == 4
