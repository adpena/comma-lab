"""Tests for experiments/probe_ot_flipweighted_mass_288.py (Sweep Arm A $0 probe).

Verifies the closed-form Menon offsets, annulus-vs-bulk prior extraction, curve interp
(in/out of grid), the validation gate, and the surface-minimised-at-zero verdict, on
constructed synthetic inputs. Behavior-verifying.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "probe_ot_288", _ROOT / "experiments" / "probe_ot_flipweighted_mass_288.py")
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)


def test_menon_uniform_priors_zero_offsets():
    b = _MOD.menon_offsets(np.array([1.0, 1.0, 1.0, 1.0, 1.0]), tau=1.0)
    assert np.allclose(b, 0.0, atol=1e-9)


def test_menon_rare_class_gets_large_positive_offset():
    # class 1 rare -> larger positive offset than the common classes (enlarge its argmax cell).
    b = _MOD.menon_offsets(np.array([10.0, 0.1, 10.0, 10.0, 10.0]), tau=1.0)
    assert b[1] == max(b)
    assert b[1] > 0.0


def test_menon_zero_sum():
    b = _MOD.menon_offsets(np.array([3.0, 1.0, 5.0, 2.0, 4.0]), tau=1.3)
    assert abs(float(b.sum())) < 1e-9


def test_curve_interp_in_and_out_of_grid():
    curve = [{"offset": -0.4, "d_seg": 0.004}, {"offset": 0.0, "d_seg": 0.003},
             {"offset": 0.4, "d_seg": 0.005}]
    assert abs(_MOD._curve_interp(curve, 0.0) - 0.003) < 1e-12
    assert abs(_MOD._curve_interp(curve, -0.2) - 0.0035) < 1e-9   # linear interp
    assert _MOD._curve_interp(curve, 1.5) is None                 # out of grid


def test_class_priors_annulus_vs_bulk():
    ls = np.array([[0, 1, 2], [3, 4, 0]], np.int64)
    mg = np.array([[0.1, 5.0, 0.2], [5.0, 0.3, 5.0]], np.float32)  # annulus = margin<1
    bulk = _MOD.class_priors(ls, None)
    ann = _MOD.class_priors(ls, mg < 1.0)
    assert bulk.tolist() == [2, 1, 1, 1, 1]        # full-frame counts
    assert ann.tolist() == [1, 0, 1, 0, 1]         # only margin<1 pixels: (0,0),(0,2)=class0/2,(1,1)=class4


def _synth_gt(tmp: Path, P=3, H=16, W=16) -> Path:
    rng = np.random.default_rng(7)
    ls = rng.integers(0, 5, (P, H, W)).astype(np.int64)
    mg = rng.uniform(0.0, 3.0, (P, H, W)).astype(np.float32)
    p = tmp / "gt.npz"
    np.savez(p, lstars=ls, margins=mg, n_pairs=np.int64(P))
    return p


def _synth_ot_result(tmp: Path, priors) -> Path:
    # Build an ot_result whose menon_analytic offsets EXACTLY equal the closed form on `priors`,
    # so the validation gate passes. Flat curves minimised at 0.
    b = _MOD.menon_offsets(np.asarray(priors), tau=1.0)
    curves = {str(c): [{"offset": o, "d_seg": 0.003 + 0.0001 * abs(o)}
                       for o in (-0.4, -0.2, 0.0, 0.2, 0.4)] for c in (0, 1, 3)}
    res = {
        "baseline_d_seg": 0.003,
        "per_class_1d_curves": curves,
        "menon_analytic": {"offsets": {str(c): float(b[c]) for c in range(5)}, "d_seg": 0.0033},
    }
    p = tmp / "ot_result.json"
    p.write_text(json.dumps(res))
    return p


def test_validation_gate_passes_when_menon_matches(tmp_path):
    gt = _synth_gt(tmp_path)
    # bulk priors of the synth gt drive the ot_result's menon so reproduction is exact.
    zc = np.load(gt)
    bulk = np.bincount(zc["lstars"].reshape(-1), minlength=5).astype(float)
    otp = _synth_ot_result(tmp_path, bulk)
    out = _MOD.run_probe(gt, otp, tau=1.0)
    assert out["validation_reproduces_measured_menon"] is True
    assert out["bulk_menon_offset_reproduction_max_abs_err"] < 1e-6
    # surface minimised at zero verdict when offsets are large / out of the flat grid
    assert out["verdict"] == "PREDICTED_NOGO_SURFACE_MINIMISED_AT_ZERO"


def test_validation_gate_fails_routes(tmp_path):
    gt = _synth_gt(tmp_path)
    # ot_result menon offsets DON'T match closed form on bulk -> validation fails -> ROUTE.
    res = {
        "baseline_d_seg": 0.003,
        "per_class_1d_curves": {str(c): [{"offset": o, "d_seg": 0.003}
                                         for o in (-0.4, 0.0, 0.4)] for c in (0, 1, 3)},
        "menon_analytic": {"offsets": {str(c): 99.0 for c in range(5)}, "d_seg": 0.0033},
    }
    otp = tmp_path / "ot_bad.json"
    otp.write_text(json.dumps(res))
    out = _MOD.run_probe(gt, otp, tau=1.0)
    assert out["verdict"] == "ROUTE_VALIDATION_FAILED"
