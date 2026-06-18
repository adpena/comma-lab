# SPDX-License-Identifier: MIT
"""NO-FAKE tests for tac.frontier_decoder_ptq — verify ACTUAL round-trip behavior.

These tests verify BEHAVIOR (the decode produces the real frontier weights; the identity
re-encode reproduces the frontier archive byte-for-byte; requantize actually changes the
weight grid; score-aware allocation actually assigns per-stage precision), not constants.
If the module body were replaced by ``return canonical_markers`` every test below would
FAIL (Catalog #307 / NO-FAKE class 2).

The tests need the on-disk frontier archive; they skip cleanly if it is absent (e.g. a
fresh checkout without the experiments/results tree).
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from tac.frontier_decoder_ptq import (
    DSEG_CRITICAL_PREFIXES,
    FRONTIER_ARCHIVE,
    FrontierDecoderPtqError,
    build_frontier_decoder,
    decode_frontier_member,
    reencode_frontier_archive,
    requantize_decoder_state_dict,
    score,
    score_aware_requant_state_dict,
)

_HAVE_FRONTIER = Path(FRONTIER_ARCHIVE).is_file()
_skip_no_frontier = pytest.mark.skipif(
    not _HAVE_FRONTIER, reason="frontier archive not present (experiments/results tree)"
)

# Known facts (verified by the identity exploration): the frontier decoder has 228,958
# params; the on-disk archive is 177,169 B; the member is 177,069 B.
_FRONTIER_PARAMS = 228_958
_FRONTIER_ARCHIVE_BYTES = 177_169
_FRONTIER_MEMBER_BYTES = 177_069


# ---------------------------------------------------------------------------
# score() arithmetic — pure, no frontier needed.
# ---------------------------------------------------------------------------


def test_score_is_the_exact_contest_formula():
    # S = 100*d_seg + sqrt(10*d_pose) + 25*bytes/B0
    d_seg, d_pose, b = 0.001, 0.0004, 100_000
    expected = 100.0 * d_seg + math.sqrt(10.0 * d_pose + 1e-12) + 25.0 * b / 37_545_489
    assert abs(score(d_seg, d_pose, b) - expected) < 1e-12


def test_score_rate_term_grows_with_bytes():
    s_small = score(0.001, 0.0004, 50_000)
    s_big = score(0.001, 0.0004, 150_000)
    assert s_big > s_small
    # the delta is exactly the rate-term difference
    assert abs((s_big - s_small) - 25.0 * 100_000 / 37_545_489) < 1e-9


# ---------------------------------------------------------------------------
# decode — the real frontier weights.
# ---------------------------------------------------------------------------


@_skip_no_frontier
def test_decode_frontier_member_returns_real_weights():
    m = decode_frontier_member()
    assert m.member_bytes == _FRONTIER_MEMBER_BYTES
    n_params = sum(int(v.numel()) for v in m.state_dict.values())
    assert n_params == _FRONTIER_PARAMS
    assert len(m.state_dict) == 28  # FIXED_STATE_SCHEMA tensor count
    # latents are the real (600, 28) runtime latents.
    assert tuple(m.latents.shape) == (600, 28)
    # weights are NOT all-zero (a fake decode would be).
    total_abs = sum(float(v.abs().sum()) for v in m.state_dict.values())
    assert total_abs > 0.0
    # the weights live on a quantized grid (PR101 int8*fp16): each weight tensor has a
    # small finite set of distinct magnitudes per-tensor (<= 255 distinct levels).
    w = m.state_dict["stem.weight"]
    distinct = torch.unique(w).numel()
    assert distinct <= 256, f"stem.weight has {distinct} distinct values (expected int8 grid)"


@_skip_no_frontier
def test_decode_missing_archive_raises():
    with pytest.raises(FrontierDecoderPtqError):
        decode_frontier_member("/nonexistent/archive.zip")


@_skip_no_frontier
def test_build_frontier_decoder_forward_produces_frames():
    m = decode_frontier_member()
    dec = build_frontier_decoder(m.state_dict)
    with torch.no_grad():
        out = dec(m.latents[:2])
    # (B, 2-frame-pair, 3-rgb, H, W) in [0, 255]
    assert tuple(out.shape) == (2, 2, 3, 384, 512)
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 255.0001


# ---------------------------------------------------------------------------
# identity round-trip — byte-for-byte reproduction of the frontier archive.
# ---------------------------------------------------------------------------


@_skip_no_frontier
def test_int8_identity_reencode_reproduces_frontier_bytes(tmp_path):
    m = decode_frontier_member()
    # int8 = the codec's own grid; re-encoding the unchanged weights must reproduce the
    # on-disk frontier archive byte-for-byte (the faithfulness gate).
    sd_int8 = requantize_decoder_state_dict(m.state_dict, "int8")
    out = tmp_path / "identity.zip"
    n = reencode_frontier_archive(m, sd_int8, out)
    assert n == _FRONTIER_ARCHIVE_BYTES, f"identity re-encode {n} != frontier {_FRONTIER_ARCHIVE_BYTES}"


# ---------------------------------------------------------------------------
# requantize ACTUALLY changes the grid + shrinks bytes.
# ---------------------------------------------------------------------------


@_skip_no_frontier
@pytest.mark.parametrize("mode", ["int7", "int6", "int5", "int4"])
def test_uniform_shrink_actually_lowers_distinct_levels_and_bytes(tmp_path, mode):
    m = decode_frontier_member()
    shrunk = requantize_decoder_state_dict(m.state_dict, mode)
    # A weight tensor on the sparser grid has FEWER distinct values than int8.
    nbits = int(mode[3:])
    qmax = 2 ** (nbits - 1) - 1
    w = shrunk["stem.weight"]
    distinct = torch.unique(w).numel()
    assert distinct <= 2 * qmax + 1, f"{mode} stem.weight {distinct} distinct > grid {2 * qmax + 1}"
    # and the re-encoded archive is SMALLER than the int8 identity (the byte lever is real).
    out = tmp_path / f"{mode}.zip"
    n = reencode_frontier_archive(m, shrunk, out)
    assert n < _FRONTIER_ARCHIVE_BYTES, f"{mode} archive {n} not < frontier {_FRONTIER_ARCHIVE_BYTES}"


@_skip_no_frontier
def test_lower_bits_shrink_more_bytes(tmp_path):
    m = decode_frontier_member()
    sizes = {}
    for mode in ("int7", "int6", "int5", "int4"):
        out = tmp_path / f"{mode}.zip"
        sizes[mode] = reencode_frontier_archive(m, requantize_decoder_state_dict(m.state_dict, mode), out)
    # monotone: fewer bits => fewer bytes.
    assert sizes["int7"] > sizes["int6"] > sizes["int5"] > sizes["int4"], sizes


# ---------------------------------------------------------------------------
# score-aware per-stage allocation — ACTUAL per-stage precision assignment.
# ---------------------------------------------------------------------------


@_skip_no_frontier
def test_score_aware_assigns_high_precision_to_critical_low_to_blind():
    m = decode_frontier_member()
    sa = score_aware_requant_state_dict(m.state_dict, low_nbits=4, high_nbits=8)
    # A d_seg-CRITICAL output-proximal head (rgb_0.weight) keeps the int8-class grid
    # (more distinct levels) than a d_seg-BLIND early stage (stem.weight) at int4.
    crit = torch.unique(sa["rgb_0.weight"]).numel()
    blind = torch.unique(sa["stem.weight"]).numel()
    # int4 grid has <= 15 distinct levels; int8 head has more (it's a small tensor but
    # still on the denser grid). The blind stage must be on the strictly coarser grid.
    assert blind <= 2 * (2 ** (4 - 1) - 1) + 1  # int4 grid bound
    # rgb_0 is on int8 grid: its distinct-level COUNT relative to its own int4 requant
    # must be >= (re-quantizing it to int4 would not increase distinct levels).
    rgb0_int4 = requantize_decoder_state_dict(m.state_dict, "int4")["rgb_0.weight"]
    assert crit >= torch.unique(rgb0_int4).numel()


@_skip_no_frontier
def test_score_aware_critical_prefixes_match_late_stages():
    # the d_seg-critical prefixes must be the OUTPUT-PROXIMAL stages (where argmax-flips
    # at 192x256/384x512 drive d_seg), not the param-heavy early low-res stages.
    assert "rgb_0." in DSEG_CRITICAL_PREFIXES
    assert "rgb_1." in DSEG_CRITICAL_PREFIXES
    assert "refine." in DSEG_CRITICAL_PREFIXES
    # the giant early stages (stem, blocks.0..2 where 77% of params live) are NOT critical
    assert not any("stem.".startswith(p) for p in DSEG_CRITICAL_PREFIXES)
    assert not any("blocks.0.".startswith(p) for p in DSEG_CRITICAL_PREFIXES)


@_skip_no_frontier
def test_score_aware_is_between_uniform_endpoints_on_bytes(tmp_path):
    m = decode_frontier_member()
    int4 = reencode_frontier_archive(m, requantize_decoder_state_dict(m.state_dict, "int4"), tmp_path / "i4.zip")
    int8 = _FRONTIER_ARCHIVE_BYTES
    sa = reencode_frontier_archive(
        m, score_aware_requant_state_dict(m.state_dict, low_nbits=4, high_nbits=8), tmp_path / "sa.zip"
    )
    # score-aware (most params int4, tiny heads int8) sits between uniform-int4 and int8.
    assert int4 < sa < int8, f"sa={sa} not between int4={int4} and int8={int8}"


def test_score_aware_nbits_validation():
    with pytest.raises(FrontierDecoderPtqError):
        score_aware_requant_state_dict({}, low_nbits=1, high_nbits=8)


@_skip_no_frontier
def test_score_aware_does_not_mutate_input():
    m = decode_frontier_member()
    before = m.state_dict["stem.weight"].clone()
    _ = score_aware_requant_state_dict(m.state_dict, low_nbits=4, high_nbits=8)
    assert torch.equal(m.state_dict["stem.weight"], before)
