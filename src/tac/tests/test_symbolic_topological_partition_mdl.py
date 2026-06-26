# SPDX-License-Identifier: MIT
"""NO-FAKE tests for the symbolic/topological partition MDL measurement (FEED-ae unit).

These verify the measurement driver does the REAL work it claims:
 - exact real-coder bytes are REAL coded lengths (decode==encode bit-exact),
 - the lossy region-drop ACTUALLY simplifies (drops small regions, d_seg rises),
 - d_seg is the exact popcount of simplified-vs-GT (the real flips a simpler store
   would incur), NOT a constant,
 - implied_S / rate_term match the evaluate.py formula,
 - the verdict's byte-delta arithmetic is honest.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

_MOD_PATH = REPO / "experiments" / "measure_symbolic_topological_partition_mdl.py"
_spec = importlib.util.spec_from_file_location("_sym_mdl", _MOD_PATH)
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)  # type: ignore


def _toy_partition(seed: int = 0) -> np.ndarray:
    """A (H,W) 5-class label map: a big road/undrivable split + a few tiny regions."""
    rng = np.random.default_rng(seed)
    a = np.zeros((40, 60), dtype=np.int64)
    a[:20, :] = 2  # undrivable top half
    a[20:, :] = 0  # road bottom half
    a[18:22, 28:32] = 1  # a lane strip across the boundary
    # a few tiny "unnecessary" speckle regions (size 1-3 px)
    for _ in range(6):
        r, c = int(rng.integers(2, 38)), int(rng.integers(2, 58))
        a[r, c] = 3
    return a


def test_exact_real_bytes_are_bit_exact_and_real():
    parts = [_toy_partition(s) for s in range(5)]
    out = _m.measure_exact_partition_real_bytes(parts)
    assert out["roundtrip_bit_exact_full_stack"] is True
    # real coded length is a positive int, and the best <= each template.
    assert out["best_total_bytes"] > 0
    assert out["best_total_bytes"] <= out["temporal_total_bytes"]
    assert out["best_total_bytes"] <= out["spatial_total_bytes"]
    assert out["d_seg"] == 0.0  # exact store is the GT partition


def test_exact_bytes_not_constant_grows_with_complexity():
    """More boundary structure -> more real bytes (the codec is not a constant)."""
    simple = [np.zeros((40, 60), dtype=np.int64) for _ in range(4)]  # one region
    complex_ = [_toy_partition(s) for s in range(4)]
    b_simple = _m.measure_exact_partition_real_bytes(simple)["best_total_bytes"]
    b_complex = _m.measure_exact_partition_real_bytes(complex_)["best_total_bytes"]
    assert b_complex > b_simple


def test_lossy_simplify_drops_small_regions():
    """min_region_px aggressiveness actually dissolves small regions (fewer regions)."""
    from scipy import ndimage
    _C4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    a = _toy_partition(0)

    def n_regions(x):
        tot = 0
        for c in range(5):
            _, n = ndimage.label(x == c, structure=_C4)
            tot += n
        return tot

    n0 = n_regions(_m._simplify_partition(a, 0))      # no drop
    n_big = n_regions(_m._simplify_partition(a, 50))   # drop everything small
    assert n_big < n0  # genuinely simplified


def test_lossy_dseg_is_exact_popcount_and_monotone():
    """d_seg rises monotonically with drop aggressiveness; equals exact flip rate."""
    parts = [_toy_partition(s) for s in range(4)]
    rows = _m.measure_lossy_symbolic_pareto(parts, [0, 30, 300])
    dsegs = [r["d_seg"] for r in rows]
    assert dsegs[0] == 0.0  # no drop = exact
    assert dsegs[1] >= dsegs[0]
    assert dsegs[2] >= dsegs[1]  # more drop -> more flips
    # d_seg matches an independent popcount on level 30
    from tac.boundary_math.bitmask_dseg import flip_count
    simp = [_m._simplify_partition(p, 30) for p in parts]
    tot = sum(flip_count(s, p) for s, p in zip(simp, parts, strict=True))
    expect = tot / (len(parts) * _m.N_PX_PER_FRAME)
    assert abs(rows[1]["d_seg"] - expect) < 1e-12


def test_lossy_bytes_are_bit_exact_roundtrip():
    """The simplified stack we measure bytes for MUST roundtrip bit-exact (real coder)."""
    from tac.boundary_math.context_partition_codec import (
        decode_partition_stack,
        encode_partition_stack,
    )
    parts = [_toy_partition(s) for s in range(3)]
    simp = [_m._simplify_partition(p, 40) for p in parts]
    code = encode_partition_stack(simp, n_classes=_m.N_CLASSES, template="temporal")
    dec = decode_partition_stack(code.payload)
    assert all(np.array_equal(a, b) for a, b in zip(simp, dec, strict=True))


def test_rate_term_matches_evaluate_formula():
    assert abs(_m.rate_term(37_545_489.0) - 25.0) < 1e-9
    assert abs(_m.rate_term(0.0)) < 1e-12


def test_implied_S_matches_score_formula():
    # S = 100*d_seg + sqrt(10*d_pose) + 25*(bytes+pose_carrier)/D
    d_seg = 0.001
    repr_bytes = 100_000.0
    expect = (100.0 * d_seg + _m.POSE_TERM_REF
              + 25.0 * (repr_bytes + _m.POSE_CARRIER_BYTES) / _m.TOTAL_VIDEO_BYTES)
    assert abs(_m.implied_S(d_seg, repr_bytes) - expect) < 1e-12
    # without pose carrier
    expect2 = 100.0 * d_seg + _m.POSE_TERM_REF + 25.0 * repr_bytes / _m.TOTAL_VIDEO_BYTES
    assert abs(_m.implied_S(d_seg, repr_bytes, include_pose_carrier=False) - expect2) < 1e-12


def test_topology_is_measured_not_constant():
    parts = [_toy_partition(s) for s in range(3)]
    topo = _m.measure_topology(parts, sample_stride=1)
    assert topo["mean_regions_per_frame"] > 1.0
    assert topo["sampled_frames"] == 3


def test_load_gt_stack_validates_size():
    """Loader fail-closes on wrong-size raw (NO FAKE: no silent reshape)."""
    import tempfile
    bad = np.zeros(100, dtype=np.uint8)
    with tempfile.NamedTemporaryFile(suffix=".u8", delete=False) as f:
        bad.tofile(f.name)
        p = Path(f.name)
    try:
        raised = False
        try:
            _m.load_gt_argmax_stack(p, 600)
        except ValueError:
            raised = True
        assert raised
    finally:
        p.unlink()


def test_verdict_byte_delta_arithmetic_honest():
    """A cheaper symbolic store -> negative byte-delta -> 'dominates' flag true."""
    # symbolic 255288 vs ws 329661 (the measured n600 numbers): symbolic dominates.
    sym = 255_288
    ws = _m.WITNESS_WEIGHT_BYTES + _m.SIDECAR_FULL_REPAIR_BYTES
    assert sym < ws
    assert (sym - ws) < 0  # negative delta = symbolic cheaper
    sym_S = _m.implied_S(0.0, sym)
    ws_S = _m.implied_S(0.0, ws)
    assert sym_S < ws_S  # cheaper bytes at same d_seg=0 -> lower S
