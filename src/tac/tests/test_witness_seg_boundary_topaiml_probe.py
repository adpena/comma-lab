# SPDX-License-Identifier: MIT
"""NO-FAKE tests for the witness seg-boundary TOP-AIML probe.

These tests verify the probe's REAL behavior, not metadata constants:

  1. The residual codec round-trips REAL bytes bit-exactly (encode -> decode == identity),
     including the order-independence (unsorted flip input) bug class the prototype's
     successor fixed, the temporal-delta chain, and the empty-set edge.
  2. The temporal-delta actually shrinks the position bytes when contours persist
     (the codec is not a no-op LZMA wrapper).
  3. The arithmetic class coder + side table round-trips the GT classes bit-exactly.
  4. INTEGRATION (real frozen SegNet, CPU, 2 pairs): the survivable subset's corrections
     ACTUALLY reduce d_seg under real re-segmentation — i.e. the coded set's pixels really
     flip toward GT on the exact eval round-trip (survival is measured, not asserted).

Per CLAUDE.md NO-FAKE: every test would FAIL if the function body were replaced by a
constant return. The codec tests assert decode(encode(x)) == x on real arrays; the
integration test asserts a real d_seg reduction from a real SegNet forward pass.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[3]
_EXP = _REPO / "experiments"
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

wt = pytest.importorskip("witness_seg_boundary_topaiml_probe")

MARGIN_BINS = np.asarray([0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 1e9], dtype=np.float64)
GRID_N = wt._N_SCORED_PER_FRAME


def _position_class_map(idx, cls):
    return {int(i): int(c) for i, c in zip(idx, cls, strict=True)}


def test_codec_roundtrip_sorted_input():
    rng = np.random.default_rng(0)
    K = 250
    flip_idx = np.sort(rng.choice(GRID_N, size=K, replace=False))
    cls = rng.integers(0, 5, size=K)
    margin = rng.random(GRID_N) * 0.6
    code, _bm = wt.encode_pair_residual(flip_idx, cls, margin, None, margin_bins=MARGIN_BINS)
    di, dc, _ = wt.decode_pair_residual(code.payload, margin, None, margin_bins=MARGIN_BINS)
    assert np.array_equal(np.sort(di), np.sort(flip_idx))
    truth = _position_class_map(flip_idx, cls)
    assert all(truth[int(i)] == int(c) for i, c in zip(di, dc, strict=True))


def test_codec_roundtrip_UNSORTED_input_order_independence():
    """The prototype-successor bug class: unsorted flip_idx must still map class->position."""
    rng = np.random.default_rng(1)
    K = 120
    flip_idx = rng.choice(GRID_N, size=K, replace=False)  # NOT sorted
    cls = rng.integers(0, 5, size=K)
    margin = rng.random(GRID_N) * 0.6
    code, _bm = wt.encode_pair_residual(flip_idx, cls, margin, None, margin_bins=MARGIN_BINS)
    di, dc, _ = wt.decode_pair_residual(code.payload, margin, None, margin_bins=MARGIN_BINS)
    assert np.array_equal(np.sort(di), np.sort(flip_idx))
    truth = _position_class_map(flip_idx, cls)
    # would FAIL if classes were misaligned to sorted positions (the fixed bug)
    assert all(truth[int(i)] == int(c) for i, c in zip(di, dc, strict=True))


def test_codec_temporal_delta_chain_roundtrip():
    rng = np.random.default_rng(2)
    margin = rng.random(GRID_N) * 0.6
    idx0 = np.sort(rng.choice(GRID_N, size=300, replace=False))
    cls0 = rng.integers(0, 5, size=300)
    code0, bm0 = wt.encode_pair_residual(idx0, cls0, margin, None, margin_bins=MARGIN_BINS)
    di0, dc0, dbm0 = wt.decode_pair_residual(code0.payload, margin, None, margin_bins=MARGIN_BINS)
    assert np.array_equal(dbm0, bm0)  # decoder reconstructs the same bitmask state

    # pair 1 shares most positions (persistent contour) -> temporal delta
    keep = idx0[20:]
    new = rng.choice(GRID_N, size=20, replace=False)
    idx1 = np.unique(np.concatenate([keep, new]))
    cls1 = rng.integers(0, 5, size=len(idx1))
    code1, bm1 = wt.encode_pair_residual(idx1, cls1, margin, bm0, margin_bins=MARGIN_BINS)
    di1, dc1, _ = wt.decode_pair_residual(code1.payload, margin, dbm0, margin_bins=MARGIN_BINS)
    assert np.array_equal(np.sort(di1), np.sort(idx1))
    truth1 = _position_class_map(idx1, cls1)
    assert all(truth1[int(i)] == int(c) for i, c in zip(di1, dc1, strict=True))


def test_temporal_delta_actually_shrinks_persistent_contours():
    """The temporal-delta is NOT a no-op: a near-identical next pair must cost FEWER position
    bytes than coding it fresh (XOR-delta is sparse where contours persist)."""
    rng = np.random.default_rng(3)
    margin = rng.random(GRID_N) * 0.6
    idx0 = np.sort(rng.choice(GRID_N, size=400, replace=False))
    cls0 = rng.integers(0, 5, size=400)
    _c0, bm0 = wt.encode_pair_residual(idx0, cls0, margin, None, margin_bins=MARGIN_BINS)

    # next pair = identical positions (perfectly persistent contour)
    code_delta, _ = wt.encode_pair_residual(idx0, cls0, margin, bm0, margin_bins=MARGIN_BINS)
    code_fresh, _ = wt.encode_pair_residual(idx0, cls0, margin, None, margin_bins=MARGIN_BINS)
    # XOR with the identical previous bitmask is all-zero -> LZMA-RAW of zeros is tiny
    assert code_delta.bitmask_bytes < code_fresh.bitmask_bytes


def test_codec_empty_set_roundtrip():
    rng = np.random.default_rng(4)
    margin = rng.random(GRID_N) * 0.6
    code, bm = wt.encode_pair_residual(
        np.zeros(0, np.int64), np.zeros(0, np.int64), margin, None, margin_bins=MARGIN_BINS
    )
    di, dc, _ = wt.decode_pair_residual(code.payload, margin, None, margin_bins=MARGIN_BINS)
    assert len(di) == 0 and len(dc) == 0


def test_class_coder_uses_real_arithmetic_coding_not_fixed_size():
    """The class stream must be arithmetic-coded (a low-entropy class distribution costs
    FEWER bytes than a high-entropy one), proving it is not a fixed-width dump."""
    rng = np.random.default_rng(5)
    K = 500
    flip_idx = np.sort(rng.choice(GRID_N, size=K, replace=False))
    margin = rng.random(GRID_N) * 0.6
    # low-entropy classes (all class 1) vs high-entropy (uniform over 5)
    cls_low = np.ones(K, dtype=np.int64)
    cls_high = rng.integers(0, 5, size=K)
    code_low, _ = wt.encode_pair_residual(flip_idx, cls_low, margin, None, margin_bins=MARGIN_BINS)
    code_high, _ = wt.encode_pair_residual(flip_idx, cls_high, margin, None, margin_bins=MARGIN_BINS)
    assert code_low.class_bytes < code_high.class_bytes


@pytest.mark.slow
def test_integration_survivable_set_really_reduces_dseg_under_real_segnet():
    """INTEGRATION (real frozen SegNet, CPU): the survivable corrections ACTUALLY flip pixels
    toward GT on the exact eval round-trip. We re-run the real SegNet on the corrected frame
    and assert the survivable pixels' new argmax == GT (real survival, not asserted)."""
    ckpt = _REPO / "experiments/results/torch_vehicle_full_mps_basin_bc20_n600/best"
    video = _REPO / "upstream/videos/0.mkv"
    if not ckpt.exists() or not video.exists():
        pytest.skip("basin checkpoint or video not present")

    import torch

    from tac.score_aware_loop.targets import load_frozen_distortion_net
    from tac.torch_vehicle.scorer_context import RealScorerContext
    from tac.torch_vehicle.vendored_imports import import_vendored

    torch.set_num_threads(2)
    score_mod = import_vendored("score")
    net = load_frozen_distortion_net(device="cpu")
    dec, latents = wt._load_basin_decoder(ckpt, "ema")
    ctx = RealScorerContext(
        str(video), device="cpu", max_pairs=2, targets_cache=".omx/tmp/witness_topaiml_test_targets"
    )
    gt_argmax = ctx.seg_targets_hard.cpu().numpy()
    n = min(2, gt_argmax.shape[0])
    idx = torch.arange(n)
    seg_out, decoded = wt._render_and_segforward(dec, net, score_mod, latents, idx)
    margin = wt._margin_map(seg_out).cpu().numpy()
    rendered_argmax = seg_out.argmax(dim=1).cpu().numpy()

    any_survivors = False
    for j in range(n):
        g, r, m = gt_argmax[j], rendered_argmax[j], margin[j]
        surv_idx, surv_cls, corr_dseg, base_dseg, n_cand = wt._survival_first_correct(
            net, score_mod, decoded[j], r, g, m, tau=0.5, max_candidates=512
        )
        # base: all surv_idx positions were flips (r != g)
        assert all(r.reshape(-1)[i] != g.reshape(-1)[i] for i in surv_idx[:20])
        # survivable: the corrected, round-tripped frame's argmax == GT at those positions.
        # That is EXACTLY the survival criterion the probe measured -> surv_cls == GT class.
        assert np.array_equal(surv_cls, g.reshape(-1)[surv_idx])
        if len(surv_idx) > 0:
            any_survivors = True
            # the coded d_seg on the candidate set is strictly below the base (survivors fixed)
            assert corr_dseg < base_dseg
    assert any_survivors, "expected at least one survivable boundary flip across 2 pairs"
