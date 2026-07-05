"""DE#3 clean warm-start-from-WEIGHTS tests (poisoned-resume-trap cure).

The v5 preserved gold (`v5_dseg0026_preserved_20260705`) has a CLEAN ema_BEST (d_seg 0.025) but a
POISONED `levelset_resume_state.npz` (__resume_epoch=150 = the deadlock, stale ep150 optimizer
moments, a frozen runaway spike-guard window). Resuming the resume_state re-enters the deadlock;
warm-starting from the WEIGHTS does not. Two mechanisms are covered:

  1. `_load_resume_state` already loads a DEPLOY ema/BEST npz (unprefixed keys) as LIVE weights with
     `has_opt=False` (fresh AdamW) + epoch=__epoch -> the deploy-npz path is ALREADY a clean
     weights-only warm-start (no new code needed for it).
  2. `_resolve_weights_only_warm_start` (the DE#3 `--warm-start-weights-only` helper) forces the
     weights-only effects EVEN from a FULL sidecar (discard moments, clear spike guard, allow lever
     drift, optional epoch override) so a warm-start is safe from the poisoned resume_state too.
     Default OFF => the resume dict is byte-identical (the moments/epoch are preserved).

MLX-free: only the pure resume/warm-start helpers + argparse are exercised (no GPU)."""
from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pytest

pytest.importorskip("mlx", reason="level-set witness trainer requires mlx")

_REPO = pathlib.Path(__file__).resolve().parents[3]
_MODPATH = _REPO / "experiments" / "train_levelset_witness_realized_through_R_mlx.py"


def _load_mod():
    if not _MODPATH.exists():
        pytest.skip(f"trainer not found at {_MODPATH}")
    spec = importlib.util.spec_from_file_location("_levelset_warm_start_under_test", _MODPATH)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:  # pragma: no cover - env-dependent
        pytest.skip(f"could not import trainer module: {type(exc).__name__}: {exc}")
    return mod


MOD = _load_mod()


def _full_sidecar_rs():
    """A resume-state dict as `_load_resume_state` returns for a FULL (poisoned) sidecar: live+ema
    weights, restored optimizer moments, has_opt True, the DEADLOCK epoch."""
    return {
        "live": {"in_proj.weight": np.zeros((4, 4), np.float32)},
        "ema": {"in_proj.weight": np.zeros((4, 4), np.float32)},
        "opt": {"in_proj.weight.v": np.ones((4, 4), np.float32),
                "in_proj.weight.m": np.ones((4, 4), np.float32), "step": np.asarray(6837)},
        "epoch": 150, "has_opt": True,
        "cfg": {"__resume_epoch": 150, "__resume_has_opt": 1, "__recent_losses": [114.0, 113.0]},
    }


# ---- _resolve_weights_only_warm_start: the DE#3 flag helper -----------------------------------
class TestResolveWeightsOnlyWarmStart:
    def test_flag_off_is_noop_moments_preserved(self):
        """Default OFF: rs untouched (opt + has_opt preserved) => byte-identical resume path."""
        rs = _full_sidecar_rs()
        out = MOD._resolve_weights_only_warm_start(
            rs, warm_start_weights_only=False, warm_start_epoch=-1, ckpt_start_epoch=151)
        assert out["discarded_opt"] is False
        assert rs["has_opt"] is True                 # NOT mutated
        assert rs["opt"]                             # moments PRESERVED
        assert out["start_epoch"] == 151             # ckpt epoch+1 unchanged
        assert out["clear_spike_guard"] is False
        assert out["allow_lever_drift"] is False

    def test_flag_on_discards_moments(self):
        rs = _full_sidecar_rs()
        out = MOD._resolve_weights_only_warm_start(
            rs, warm_start_weights_only=True, warm_start_epoch=-1, ckpt_start_epoch=151)
        assert out["discarded_opt"] is True
        assert rs["has_opt"] is False                # fresh AdamW
        assert rs["opt"] == {}                       # moments DISCARDED

    def test_flag_on_weights_untouched(self):
        """Only moments are discarded; the trained WEIGHTS (live/ema) are preserved."""
        rs = _full_sidecar_rs()
        MOD._resolve_weights_only_warm_start(
            rs, warm_start_weights_only=True, warm_start_epoch=-1, ckpt_start_epoch=151)
        assert "in_proj.weight" in rs["live"] and "in_proj.weight" in rs["ema"]

    def test_flag_on_default_epoch_keeps_ckpt_plus_one(self):
        rs = _full_sidecar_rs()
        out = MOD._resolve_weights_only_warm_start(
            rs, warm_start_weights_only=True, warm_start_epoch=-1, ckpt_start_epoch=151)
        assert out["start_epoch"] == 151

    def test_flag_on_epoch_override_applied(self):
        """--warm-start-epoch 126 resets the start epoch off the DEADLOCK epoch (150) to just past
        the ep125 BEST verdict."""
        rs = _full_sidecar_rs()
        out = MOD._resolve_weights_only_warm_start(
            rs, warm_start_weights_only=True, warm_start_epoch=126, ckpt_start_epoch=151)
        assert out["start_epoch"] == 126

    def test_flag_on_epoch_override_zero_is_valid(self):
        rs = _full_sidecar_rs()
        out = MOD._resolve_weights_only_warm_start(
            rs, warm_start_weights_only=True, warm_start_epoch=0, ckpt_start_epoch=151)
        assert out["start_epoch"] == 0

    def test_flag_on_negative_epoch_falls_back_to_ckpt(self):
        rs = _full_sidecar_rs()
        out = MOD._resolve_weights_only_warm_start(
            rs, warm_start_weights_only=True, warm_start_epoch=-5, ckpt_start_epoch=127)
        assert out["start_epoch"] == 127

    def test_flag_on_clears_guard_and_allows_lever_drift(self):
        rs = _full_sidecar_rs()
        out = MOD._resolve_weights_only_warm_start(
            rs, warm_start_weights_only=True, warm_start_epoch=-1, ckpt_start_epoch=151)
        assert out["clear_spike_guard"] is True
        assert out["allow_lever_drift"] is True

    def test_ckpt_had_opt_reported_true_for_full_sidecar(self):
        rs = _full_sidecar_rs()
        out = MOD._resolve_weights_only_warm_start(
            rs, warm_start_weights_only=True, warm_start_epoch=-1, ckpt_start_epoch=151)
        assert out["ckpt_had_opt"] is True           # the poisoned sidecar DID carry moments

    def test_ckpt_had_opt_false_for_deploy_npz_rs(self):
        """A deploy-npz rs (no opt, has_opt False) reports ckpt_had_opt False => the flag is a no-op
        for that path (moments were already fresh)."""
        rs = {"live": {"w": np.zeros(3, np.float32)}, "ema": {}, "opt": {}, "epoch": 125,
              "has_opt": False, "cfg": {"__epoch": 125}}
        out = MOD._resolve_weights_only_warm_start(
            rs, warm_start_weights_only=True, warm_start_epoch=-1, ckpt_start_epoch=126)
        assert out["ckpt_had_opt"] is False
        assert out["start_epoch"] == 126


# ---- _load_resume_state: deploy-npz vs full-sidecar distinction --------------------------------
class TestLoadResumeStateWarmStartMechanism:
    def test_deploy_npz_fallback_is_fresh_moments(self, tmp_path):
        """A plain deploy npz (unprefixed keys + __epoch) loads as LIVE weights with has_opt=False
        (=> fresh AdamW) + ema empty + epoch=__epoch. THIS is why --resume-from ema_BEST.npz is
        already a clean weights-only warm-start."""
        p = tmp_path / "deploy_best.npz"
        np.savez(p, **{"in_proj.weight": np.ones((4, 4), np.float32),
                       "palette": np.ones((5, 3), np.float32),
                       "__epoch": np.asarray(125), "__cfg_hidden_dim": np.asarray(96)})
        rs = MOD._load_resume_state(p)
        assert rs["has_opt"] is False                 # fresh moments
        assert rs["opt"] == {}
        assert rs["ema"] == {}
        assert "in_proj.weight" in rs["live"]         # deploy weights become LIVE
        assert rs["epoch"] == 125                      # continues past ep125 (=> start_epoch 126)

    def test_full_sidecar_restores_moments_the_poisoned_path(self, tmp_path):
        """A full sidecar (liveP__/emaP__/optP__ + __resume_has_opt=1 + __resume_epoch) loads with
        has_opt True + opt populated + the DEADLOCK epoch = the poisoned-resume path this flag
        avoids."""
        p = tmp_path / "resume_state.npz"
        np.savez(p, **{"liveP__in_proj.weight": np.ones((4, 4), np.float32),
                       "emaP__in_proj.weight": np.ones((4, 4), np.float32),
                       "optP__in_proj.weight.v": np.ones((4, 4), np.float32),
                       "__resume_has_opt": np.asarray(1),
                       "__resume_epoch": np.asarray(150)})
        rs = MOD._load_resume_state(p)
        assert rs["has_opt"] is True
        assert rs["opt"]                               # moments restored (the deadlock preconditioner)
        assert rs["epoch"] == 150                       # the deadlock epoch (=> start_epoch 151)

    def test_deploy_npz_then_weights_only_helper_stays_fresh(self, tmp_path):
        """End-to-end: load a deploy npz -> already fresh -> the weights-only helper keeps it fresh
        and advances to epoch+1 (byte-identical outcome to the plain deploy-npz resume)."""
        p = tmp_path / "deploy_best.npz"
        np.savez(p, **{"in_proj.weight": np.ones((2, 2), np.float32), "__epoch": np.asarray(125)})
        rs = MOD._load_resume_state(p)
        out = MOD._resolve_weights_only_warm_start(
            rs, warm_start_weights_only=True, warm_start_epoch=-1, ckpt_start_epoch=rs["epoch"] + 1)
        assert rs["has_opt"] is False and rs["opt"] == {}
        assert out["start_epoch"] == 126


# ---- argparse contract: the flags exist + default OFF => byte-identical resume -----------------
class TestWarmStartArgparseContract:
    _SRC = _MODPATH.read_text()

    def test_weights_only_flag_declared_default_false(self):
        assert '"--warm-start-weights-only"' in self._SRC
        # BooleanOptionalAction + default=False on the same add_argument call.
        idx = self._SRC.index('"--warm-start-weights-only"')
        window = self._SRC[idx:idx + 400]
        assert "BooleanOptionalAction" in window and "default=False" in window

    def test_warm_start_epoch_flag_declared_default_neg1(self):
        assert '"--warm-start-epoch"' in self._SRC
        idx = self._SRC.index('"--warm-start-epoch"')
        window = self._SRC[idx:idx + 300]
        assert "default=-1" in window

    def test_helper_wired_at_resume_site(self):
        # the helper must actually be CALLED in the resume block (not just defined) — guards the
        # dangling-helper trap.
        assert "_resolve_weights_only_warm_start(" in self._SRC
        assert self._SRC.count("_resolve_weights_only_warm_start(") >= 2  # def + >=1 callsite
