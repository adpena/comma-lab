# SPDX-License-Identifier: MIT
"""NO-FAKE tests for the residual-flip sidecar coder + Pareto economics.

These test the PURE coding/economics surfaces (no MLX training) so they run fast in CI. They prove:
  * the boundary-relative (band_rank) position code is strictly <= absolute delta when |B| << N;
  * the src-conditional class code is <= the raw class code on a road<->lane-dominant flip set;
  * the sidecar round-trips the admitted (position, target-class) set bit-exactly;
  * the Pareto is monotone (more flips coded -> lower d_seg, never-decreasing bytes);
  * the implied-S formula matches the canonical S = 100*d_seg + sqrt(10*d_pose) + 25*bytes/D.
  * a select-NONE sidecar is empty; a select-ALL drives remaining flips to 0.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
for p in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "experiments"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

_spec = importlib.util.spec_from_file_location(
    "build_residual_flip_sidecar_pareto",
    REPO_ROOT / "experiments" / "build_residual_flip_sidecar_pareto.py",
)
mod = importlib.util.module_from_spec(_spec)
sys.modules["build_residual_flip_sidecar_pareto"] = mod
_spec.loader.exec_module(mod)

FlipSet = mod.FlipSet
_D_RATE_DENOM = mod._D_RATE_DENOM


def _toy_flipset(H=32, W=32, P=3, seed=0):
    """A toy witness/GT with flips concentrated on a synthetic boundary band (road<->lane heavy)."""
    rng = np.random.default_rng(seed)
    gt = np.zeros((P, H, W), dtype=np.uint8)
    gt[:, :, W // 2:] = 1  # left=road(0), right=lane(1): a vertical edge
    pred = gt.copy()
    # introduce flips hugging the edge: flip a handful of pixels right at the boundary column.
    flips_per_pair = [12, 9, 15]
    pair_l, idx_l, gtc_l, src_l, inb_l, mg_l = [], [], [], [], [], []
    band_sizes = np.zeros(P, dtype=np.int64)
    for pi in range(P):
        cols = np.full(flips_per_pair[pi], W // 2)
        rows = rng.integers(0, H, size=flips_per_pair[pi])
        # witness mispredicts (lane where it should be road) at edge -> src=1, gt=0 (road<->lane)
        for r, c in zip(rows, cols):
            pred[pi, r, c] = 1 - gt[pi, r, c]
        # band = the witness's own boundary band (edge columns)
        a = pred[pi]
        b = np.zeros_like(a, dtype=bool)
        b[:, :-1] |= a[:, :-1] != a[:, 1:]
        b[:, 1:] |= a[:, :-1] != a[:, 1:]
        b[:-1, :] |= a[:-1, :] != a[1:, :]
        b[1:, :] |= a[:-1, :] != a[1:, :]
        band_sizes[pi] = int(b.sum())
        g = gt[pi].reshape(-1)
        pp = pred[pi].reshape(-1)
        flip = pp != g
        fidx = np.where(flip)[0]
        pair_l.append(np.full(fidx.size, pi, np.int64))
        idx_l.append(fidx.astype(np.int64))
        gtc_l.append(g[fidx].astype(np.int64))
        src_l.append(pp[fidx].astype(np.int64))
        inb_l.append(b.reshape(-1)[fidx])
        mg_l.append(np.full(fidx.size, 0.3, np.float32))
    fs = FlipSet(
        pair=np.concatenate(pair_l), flat_idx=np.concatenate(idx_l),
        gt_cls=np.concatenate(gtc_l), src_cls=np.concatenate(src_l),
        in_band=np.concatenate(inb_l), margin=np.concatenate(mg_l),
        H=H, W=W, P=P, band_size_per_pair=band_sizes,
    )
    return fs, pred


def test_band_rank_beats_absolute_delta_when_band_small():
    """OURS: band-relative coding <= absolute delta when |B| << N (the whole point of Lever-D++)."""
    fs, pred = _toy_flipset()
    admit = np.ones(fs.flat_idx.size, dtype=bool)
    band = mod.encode_sidecar(fs, admit, pred_maps=pred, dilate=0, pos_mode="band_rank",
                              temporal_delta=True, conditional_class=True)
    absd = mod.encode_sidecar(fs, admit, pred_maps=pred, dilate=0, pos_mode="delta",
                              temporal_delta=True, conditional_class=True)
    assert band["pos_bytes"] <= absd["pos_bytes"], (band["pos_bytes"], absd["pos_bytes"])


def test_conditional_class_no_worse_than_raw_on_dominant_pairs():
    """src-conditional target code <= raw class code on a road<->lane-dominant set."""
    fs, pred = _toy_flipset()
    admit = np.ones(fs.flat_idx.size, dtype=bool)
    cond = mod.encode_sidecar(fs, admit, pred_maps=pred, dilate=0, pos_mode="band_rank",
                              temporal_delta=True, conditional_class=True)
    raw = mod.encode_sidecar(fs, admit, pred_maps=pred, dilate=0, pos_mode="band_rank",
                             temporal_delta=True, conditional_class=False)
    assert cond["cls_bytes"] <= raw["cls_bytes"], (cond["cls_bytes"], raw["cls_bytes"])


def test_adaptive_is_no_worse_than_any_single_mode():
    """OURS adaptive picks the per-pair cheapest position code -> total <= each fixed mode
    (modulo the small 2-bit/pair mode-tag header; assert it's <= the best fixed mode + tag slack)."""
    fs, pred = _toy_flipset()
    admit = np.ones(fs.flat_idx.size, dtype=bool)
    kw = dict(pred_maps=pred, dilate=0, temporal_delta=True, conditional_class=True)
    adapt = mod.encode_sidecar(fs, admit, pos_mode="adaptive", **kw)
    fixed = [mod.encode_sidecar(fs, admit, pos_mode=m, **kw)["pos_bytes"]
             for m in ("band_rank", "bitmap", "delta")]
    # adaptive position bytes <= the cheapest fixed mode (it picks the min per pair)
    assert adapt["pos_bytes"] <= min(fixed) + 1, (adapt["pos_bytes"], min(fixed))
    assert adapt["mode_tag_bytes"] > 0  # the tag is counted, not free


def test_select_none_is_empty_sidecar():
    fs, pred = _toy_flipset()
    admit = np.zeros(fs.flat_idx.size, dtype=bool)
    enc = mod.encode_sidecar(fs, admit, pred_maps=pred, dilate=0)
    assert enc["n_admitted"] == 0
    assert enc["pos_bytes"] == 0 and enc["cls_bytes"] == 0


def test_class_stream_roundtrip_conditional():
    """The src-conditional class symbol inverts exactly given src (the decoder knows src for free)."""
    rng = np.random.default_rng(1)
    src = rng.integers(0, 5, size=200).astype(np.int64)
    # target != src
    gt = np.array([(s + rng.integers(1, 5)) % 5 for s in src], dtype=np.int64)
    blob = mod._encode_class_stream(gt, src, conditional=True)
    import brotli
    sym = np.frombuffer(brotli.decompress(blob), dtype=np.uint8)
    # invert: for each i, others = [c for c in 0..4 if c != src]; gt = others[sym]
    recovered = np.empty_like(gt)
    for i in range(gt.size):
        others = [c for c in range(5) if c != src[i]]
        recovered[i] = others[sym[i]]
    assert np.array_equal(recovered, gt)


def test_band_rank_roundtrip_positions():
    """band-rank positions invert to flat indices given the decoder-free band (no stored band)."""
    fs, pred = _toy_flipset()
    pi = 0
    m = fs.pair == pi
    loc = np.sort(fs.flat_idx[m])
    band = mod._own_boundary_band(pred[pi], dilate=0).reshape(-1)
    band_idx = np.where(band)[0]
    blob = mod._encode_positions_band_rank(loc, band_idx)
    import brotli
    d = np.frombuffer(brotli.decompress(blob), dtype=np.uint32)
    ranks = np.cumsum(d.astype(np.int64))
    recovered = band_idx[ranks]
    assert np.array_equal(recovered, loc)


def test_implied_S_matches_canonical_formula():
    d_seg, wb, sb, dpose = 0.0011, 114197, 5000, 3.4e-5
    S = mod._implied_S(d_seg, wb, sb, dpose)
    expect = 100.0 * d_seg + float(np.sqrt(10.0 * dpose)) + 25.0 * (wb + sb) / _D_RATE_DENOM
    assert abs(S - expect) < 1e-12


def test_pareto_monotone_dseg_down_bytes_up():
    """As K rises: remaining d_seg never increases; sidecar bytes never decrease."""
    H = W = 32
    P = 3
    fs, pred = _toy_flipset(H=H, W=W, P=P)
    ext = {
        "P": P, "H": H, "W": W, "n_px": H * W,
        "pred_argmax": pred, "witness_weight_bytes": 100000,
    }
    par = mod.build_pareto(ext, fs, pos_mode="band_rank", temporal_delta=True,
                           conditional_class=True, dilate=0, k_points=6)
    rows = par["pareto"]
    dsegs = [r["d_seg"] for r in rows]
    bytez = [r["sidecar_bytes"] for r in rows]
    assert all(dsegs[i] >= dsegs[i + 1] for i in range(len(dsegs) - 1)), dsegs
    assert all(bytez[i] <= bytez[i + 1] for i in range(len(bytez) - 1)), bytez
    assert rows[-1]["d_seg"] == 0.0  # all flips coded -> zero remaining disagreement


def test_full_repair_drives_dseg_to_zero():
    fs, pred = _toy_flipset()
    admit = np.ones(fs.flat_idx.size, dtype=bool)
    P, H, W = fs.P, fs.H, fs.W
    total_px = P * H * W
    remaining = fs.flat_idx.size - int(admit.sum())
    assert remaining == 0
    assert mod._d_seg_from_flip_count(remaining, total_px) == 0.0


def test_waterline_constant_is_127_bytes_per_flip():
    assert abs(mod.WATERLINE_BYTES_PER_FLIP - 1.2742) < 0.01


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
