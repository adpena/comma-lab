"""Tests for experiments/probe_msal_reachability_orthogonality_268.py (Sweep Arm A $0 probe).

Verifies the orthogonality/redundancy/flatness verdict logic and the per-pair stats on
constructed synthetic GT caches (no heavy artifacts). Behavior-verifying, not constant-checking:
each test would FAIL if the correlation or dynamic-range guard were broken.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "probe_msal_268", _ROOT / "experiments" / "probe_msal_reachability_orthogonality_268.py")
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)


def _write_caches(tmp: Path, margins: np.ndarray, sR: np.ndarray) -> tuple[Path, Path]:
    gt = tmp / "gt.npz"
    sr = tmp / "gt_sR.npz"
    np.savez(gt, margins=margins.astype(np.float32),
             lstars=np.zeros(margins.shape, np.int64), n_pairs=np.int64(margins.shape[0]))
    np.savez(sr, sR=sR.astype(np.float32), n_pairs=np.int64(sR.shape[0]))
    return gt, sr


def test_orthogonal_when_sR_independent_and_variable(tmp_path):
    rng = np.random.default_rng(0)
    P, H, W = 4, 40, 40
    # margins: many pixels in the annulus (<band=1.0); sR: independent uniform in [0,1] (cv large).
    margins = rng.uniform(0.0, 0.9, (P, H, W)).astype(np.float32)   # all in annulus
    sR = rng.uniform(0.0, 1.0, (P, H, W)).astype(np.float32)        # independent of margin
    gt, sr = _write_caches(tmp_path, margins, sR)
    out = _MOD.run_probe(gt, sr, tau=0.5, band=1.0)
    assert out["verdict"] == "ORTHOGONAL_ADDS_SIGNAL", out
    assert abs(out["spearman_sR_vs_fragility_weight"]["mean"]) < 0.15
    assert out["sR_ann_cv"]["mean"] > 0.3


def test_redundant_when_sR_tracks_fragility(tmp_path):
    rng = np.random.default_rng(1)
    P, H, W = 4, 40, 40
    margins = rng.uniform(0.0, 0.9, (P, H, W)).astype(np.float32)
    # sR := the fragility weight itself (+ tiny noise) => strong positive Spearman => REDUNDANT.
    sal = np.exp(-margins / 0.5)
    sR = (sal + rng.normal(0, 1e-4, sal.shape)).clip(0, 1).astype(np.float32)
    gt, sr = _write_caches(tmp_path, margins, sR)
    out = _MOD.run_probe(gt, sr, tau=0.5, band=1.0)
    assert out["verdict"] == "REDUNDANT_TRACKS_FRAGILITY", out
    assert out["spearman_sR_vs_fragility_weight"]["mean"] >= 0.15


def test_uninformative_when_sR_flat(tmp_path):
    rng = np.random.default_rng(2)
    P, H, W = 4, 40, 40
    margins = rng.uniform(0.0, 0.9, (P, H, W)).astype(np.float32)
    # sR ~ constant (cv near zero): rho~0 but NO targeting info => must NOT read as ORTHOGONAL route.
    sR = np.full((P, H, W), 0.5, np.float32) + rng.normal(0, 1e-5, (P, H, W)).astype(np.float32)
    gt, sr = _write_caches(tmp_path, margins, sR)
    out = _MOD.run_probe(gt, sr, tau=0.5, band=1.0)
    assert out["verdict"] == "UNINFORMATIVE_FLAT_IN_ANNULUS", out
    assert out["sR_ann_cv"]["mean"] <= 0.3


def test_degenerate_annulus_skipped(tmp_path):
    # No pixel in the annulus (all margins large) -> pair skipped -> RuntimeError.
    P, H, W = 2, 20, 20
    margins = np.full((P, H, W), 5.0, np.float32)   # >band
    sR = np.full((P, H, W), 0.5, np.float32)
    gt, sr = _write_caches(tmp_path, margins, sR)
    with pytest.raises(RuntimeError):
        _MOD.run_probe(gt, sr, tau=0.5, band=1.0)
