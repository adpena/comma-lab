"""Hardening tests for the level-set witness EMA/checkpoint/resume machinery.

Covers the 2026-06-30 hardening (operator: "Harden and fix EMA and all checkpoint and resumable
behaviors"): the BEST-d_seg checkpoint preservation + atomic JSON pointer + the contract that a
best DEPLOY npz (EMA shadow + cfg) is warm-startable. The gap these guard: the rolling "latest" +
per-stage ckpts could DRIFT PAST the best realized d_seg (tau over-trains past its knee; l7/Muon
oscillate) -> the best EMA shadow was LOST (forced a manual ep725 snapshot worse than the ep700
best). See CLAUDE.md "EMA" + "Resumability + per-stage checkpoints" non-negotiables.

The trainer module imports mlx/torch at top; skip cleanly where unavailable."""
from __future__ import annotations

import importlib.util
import json
import pathlib

import numpy as np
import pytest

pytest.importorskip("mlx", reason="level-set witness trainer requires mlx")

_REPO = pathlib.Path(__file__).resolve().parents[3]
_MODPATH = _REPO / "experiments" / "train_levelset_witness_realized_through_R_mlx.py"


def _load_mod():
    if not _MODPATH.exists():
        pytest.skip(f"trainer not found at {_MODPATH}")
    spec = importlib.util.spec_from_file_location("_levelset_witness_trainer_under_test", _MODPATH)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)  # executes heavy imports (mlx/torch/tac); no GPU ops
    except Exception as exc:  # pragma: no cover - env-dependent
        pytest.skip(f"could not import trainer module: {type(exc).__name__}: {exc}")
    return mod


MOD = _load_mod()


# ---- _is_new_best: the NO-FAKE promotion rule (finite + strictly-better only) -----------------
class TestIsNewBest:
    def test_first_finite_beats_inf(self):
        assert MOD._is_new_best(0.0042, float("inf")) is True

    def test_strictly_better_promotes(self):
        assert MOD._is_new_best(0.0041, 0.0042) is True

    def test_worse_rejected(self):
        assert MOD._is_new_best(0.0043, 0.0042) is False

    def test_exact_tie_rejected_keeps_earlier(self):
        assert MOD._is_new_best(0.0042, 0.0042) is False

    def test_sub_ulp_improvement_rejected_no_churn(self):
        # 1e-12 guard: a float-noise "improvement" must NOT rewrite the best ckpt.
        assert MOD._is_new_best(0.0042 - 1e-13, 0.0042) is False

    def test_real_improvement_above_guard_promotes(self):
        assert MOD._is_new_best(0.0042 - 1e-6, 0.0042) is True

    def test_nan_never_wins(self):
        assert MOD._is_new_best(float("nan"), float("inf")) is False
        assert MOD._is_new_best(float("nan"), 0.0042) is False

    def test_posinf_never_wins(self):
        assert MOD._is_new_best(float("inf"), float("inf")) is False


# ---- _atomic_write_json: durable pointer, atomic, refuses /tmp -------------------------------
class TestAtomicWriteJson:
    def test_roundtrip_valid_json(self, tmp_path):
        p = tmp_path / "levelset_best.json"
        obj = {"d_seg": 0.004227, "epoch": 700, "path": "levelset_witness_ema_BEST.npz"}
        MOD._atomic_write_json(p, obj)
        got = json.loads(p.read_text())
        assert got == obj

    def test_no_tmp_leftover(self, tmp_path):
        p = tmp_path / "x.json"
        MOD._atomic_write_json(p, {"a": 1})
        leftovers = list(tmp_path.glob(".x.json.tmp.*"))
        assert leftovers == [], f"atomic write left a tmp file: {leftovers}"

    def test_overwrite_is_atomic(self, tmp_path):
        p = tmp_path / "best.json"
        MOD._atomic_write_json(p, {"d_seg": 0.005, "epoch": 100})
        MOD._atomic_write_json(p, {"d_seg": 0.004, "epoch": 200})  # new best supersedes
        assert json.loads(p.read_text())["d_seg"] == 0.004

    def test_refuses_tmp_path(self):
        with pytest.raises(ValueError):
            MOD._atomic_write_json(pathlib.Path("/tmp/should_refuse_best.json"), {"a": 1})


# ---- the BEST deploy-npz warm-start contract: a shadow-only deploy npz loads as live + epoch --
class TestBestDeployNpzIsWarmStartable:
    """The core hardening claim: the best ckpt is an EMA-shadow DEPLOY npz, and resume must seed
    `live` from it (so the next arm warm-starts) AND restore the epoch position. _load_resume_state
    maps unprefixed (deploy) keys into `live` and reads `__epoch`."""

    def test_deploy_npz_seeds_live_and_epoch(self, tmp_path):
        rng = np.random.default_rng(0)
        shadow = {
            "in_proj.weight": rng.standard_normal((8, 4)).astype(np.float32),
            "in_proj.bias": rng.standard_normal((8,)).astype(np.float32),
            "film.weight": rng.standard_normal((8, 8)).astype(np.float32),
            "code": rng.standard_normal((4, 8)).astype(np.float32),
        }
        arrays = dict(shadow)
        arrays["__epoch"] = np.asarray(700)  # the best epoch (deploy-npz provenance)
        arrays["__cfg_hidden_dim"] = np.asarray(8)
        p = tmp_path / "levelset_witness_ema_BEST.npz"
        MOD._atomic_savez(p, arrays)

        rs = MOD._load_resume_state(p)
        # deploy npz -> unprefixed shadow keys become the live-weight fallback (warm-start source)
        assert set(rs["live"].keys()) == set(shadow.keys()), "deploy npz must seed live<-shadow"
        for k, v in shadow.items():
            np.testing.assert_array_equal(rs["live"][k], v)
        # epoch restored from __epoch (resume-position-faithful)
        assert rs["epoch"] == 700
        # cfg provenance carried
        assert int(rs["cfg"]["__cfg_hidden_dim"]) == 8

    def test_resolve_resume_path_accepts_explicit_npz(self, tmp_path):
        p = tmp_path / "levelset_witness_ema_BEST.npz"
        MOD._atomic_savez(p, {"in_proj.weight": np.zeros((2, 2), np.float32), "__epoch": np.asarray(5)})
        assert MOD._resolve_resume_path(p) == p  # explicit file path returned as-is


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
