"""Tests for experiments/probe_code_effrank_cross_ckpt_140.py (Sweep Arm A $0 rate probe).

Verifies the SVD eff-rank / energy-percentile math and the per-vehicle verdict branches on
constructed synthetic checkpoints. Behavior-verifying.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "probe_code_140", _ROOT / "experiments" / "probe_code_effrank_cross_ckpt_140.py")
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)


def test_effrank_rank1_is_low():
    rng = np.random.default_rng(0)
    u = rng.normal(size=(200, 1))
    v = rng.normal(size=(1, 32))
    code = (u @ v).astype(np.float32) + 1e-3 * rng.normal(size=(200, 32)).astype(np.float32)
    r = _MOD.code_effrank(code)
    assert r["eff_rank_participation_ratio"] < 3.0
    assert r["rank_90pct_energy"] <= 3


def test_effrank_fullrank_is_high():
    rng = np.random.default_rng(1)
    code = rng.normal(size=(200, 19)).astype(np.float32)   # isotropic full-rank
    r = _MOD.code_effrank(code)
    assert r["eff_rank_participation_ratio"] > 12.0
    assert r["rank_90pct_energy"] >= 14


def test_energy_rank_monotone():
    sv = np.array([10.0, 3.0, 1.0, 0.5, 0.1])
    assert _MOD._energy_rank(sv, 0.5) <= _MOD._energy_rank(sv, 0.9) <= _MOD._energy_rank(sv, 0.99)


def _write_ckpt(tmp: Path, name: str, code: np.ndarray) -> Path:
    p = tmp / f"{name}.npz"
    np.savez(p, code=code.astype(np.float32))
    return p


def test_verdict_mod32to19_safe_but_sub19_saturated(tmp_path):
    rng = np.random.default_rng(2)
    # mod-32 checkpoint: 90% energy within 19 dims (safe fold). Build code with ~15 strong dims.
    strong = rng.normal(size=(300, 15))
    pad32 = 1e-3 * rng.normal(size=(300, 17))
    code32 = np.concatenate([strong, pad32], axis=1)
    # live v9 mod-19 checkpoint: near-full-rank 19 (saturated) => no sub-19 headroom.
    code19 = rng.normal(size=(300, 19))
    c32 = _write_ckpt(tmp_path, "mod32_x", code32)
    c19 = _write_ckpt(tmp_path, "v9_cgauge_x", code19)     # name triggers the "live" selector
    out = _MOD.run_probe([c32, c19], tmp_path)
    assert out["max_90pct_rank_mod32"] <= 19
    assert out["verdict"] == "MOD32TO19_SAFE_BUT_SUB19_SATURATED_ON_LIVE", out


def test_verdict_sub19_headroom_when_live_unsaturated(tmp_path):
    rng = np.random.default_rng(3)
    strong = rng.normal(size=(300, 12))
    code32 = np.concatenate([strong, 1e-3 * rng.normal(size=(300, 20))], axis=1)
    # live mod-19 that is near rank-1 (unsaturated) => sub-19 headroom.
    u = rng.normal(size=(300, 1)); v = rng.normal(size=(1, 19))
    code19 = u @ v + 1e-3 * rng.normal(size=(300, 19))
    c32 = _write_ckpt(tmp_path, "mod32_y", code32)
    c19 = _write_ckpt(tmp_path, "v9_cgauge_y", code19)
    out = _MOD.run_probe([c32, c19], tmp_path)
    assert out["verdict"] == "MOD32TO19_SAFE_SUB19_HEADROOM_ON_LIVE", out


def test_verdict_rate_risk_when_mod32_needs_more_than_19(tmp_path):
    rng = np.random.default_rng(4)
    # mod-32 code that genuinely needs >19 dims for 90% energy (isotropic 32) => fold risk.
    code32 = rng.normal(size=(400, 32))
    c32 = _write_ckpt(tmp_path, "mod32_z", code32)
    out = _MOD.run_probe([c32], tmp_path)
    assert out["max_90pct_rank_mod32"] > 19
    assert out["verdict"] == "MOD32TO19_RATE_RISK", out


def test_missing_and_nocode_rows(tmp_path):
    good = _write_ckpt(tmp_path, "mod32_ok", np.random.default_rng(5).normal(size=(100, 32)))
    missing = tmp_path / "nope.npz"
    out = _MOD.run_probe([good, missing], tmp_path)
    statuses = {r.get("status") for r in out["rows"]}
    assert "OK" in statuses and "MISSING" in statuses
