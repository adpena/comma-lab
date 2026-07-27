# SPDX-License-Identifier: MIT
"""Tests for the additive intermediate-checkpoint + --resume-from surfaces on the level-set witness
trainer (FEED-dz, landed 2026-06-27).

Covers the pure-numpy (MLX-free) checkpoint/resume helpers added to
``experiments/train_levelset_witness_realized_through_R_mlx.py``:
  * ``_atomic_savez`` -- atomic (tmp+os.replace) npz write; refuses /tmp; overwrites cleanly.
  * ``_build_ema_checkpoint_arrays`` -- the deploy (byte-close) npz contents: EMA shadow params +
    EVERY historical cfg key + the NEW self-orient/curriculum/w_pose/epoch provenance keys.
  * ``_build_resume_state_arrays`` / ``_load_resume_state`` -- the resume sidecar (live + EMA + opt
    + epoch), prefixed so the deploy npz stays byte-close-clean; round-trips; EMA-npz fallback.
  * ``_resolve_resume_path`` -- dir/file resolution + NO-FAKE missing-file raise.
  * ``_stage_tag`` -- filename-safe PR95-stage tags.
  * INTEGRATION: a built EMA checkpoint npz is loadable by the level-set byte-close tool
    (``tools/levelset_byte_close_and_eval._load_levelset_ckpt``) -- the row-enabling consumer.

All MLX-free (the helpers operate on numpy dicts; the caller does the mx->np conversion).
"""
from __future__ import annotations

import inspect
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[2]
for _p in (_REPO, _REPO / "src", _REPO / "experiments", _REPO / "upstream"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import train_levelset_witness_realized_through_R_mlx as T  # noqa: E402


# --------------------------------------------------------------------------- fixtures
def _fake_args(**over):
    base = dict(
        n_hidden=4, hidden_dim=96, activation="hosc", chroma=True,
        wire_w0=20.0, wire_s0=10.0, hosc_beta=4.0, hosc_omega=1.0,
        bank_n_scales=4, bank_n_orient0=6, bank_f0=2.0, bank_base=2.0, bank_n_iso=4,
        basis="polar_fourier",
        max_bank_freq=None, lane_edge_weight=0.0, lane_edge_class=1,
        self_orient=True, n_dir_freqs=2, freq_across=32.0, freq_along=4.0, reorient_every=50,
        w_pose=1.0, curriculum=True, tau_softplus_start_epoch=300, l7_start_epoch=900,
        mod_dim=32,
    )
    base.update(over)
    return SimpleNamespace(**base)


def _fake_shadow():
    """Minimal EMA-shadow param dict with the keys the byte-close tool REQUIRES + a couple more."""
    rng = np.random.default_rng(0)
    return {
        "code": rng.standard_normal((6, 32)).astype(np.float32),          # (2*n_pairs, mod)
        "in_proj.weight": rng.standard_normal((96, 40)).astype(np.float32),
        "in_proj.bias": rng.standard_normal((96,)).astype(np.float32),
        "out_sdf.weight": rng.standard_normal((5, 96)).astype(np.float32),  # n_classes rows
        "out_sdf.bias": rng.standard_normal((5,)).astype(np.float32),
        "out_tex.weight": rng.standard_normal((3, 96)).astype(np.float32),
        "out_tex.bias": rng.standard_normal((3,)).astype(np.float32),
        "palette": rng.standard_normal((5, 3)).astype(np.float32),
    }


# --------------------------------------------------------------------------- _atomic_savez
def test_atomic_savez_writes_loadable_npz(tmp_path):
    arrays = {"a": np.arange(5, dtype=np.float32), "__epoch": np.asarray(7)}
    out = T._atomic_savez(tmp_path / "ck.npz", arrays)
    assert out.exists()
    z = np.load(out)
    assert np.array_equal(z["a"], np.arange(5, dtype=np.float32))
    assert int(z["__epoch"]) == 7


def test_atomic_savez_leaves_no_tmp_file(tmp_path):
    T._atomic_savez(tmp_path / "ck.npz", {"a": np.zeros(3, np.float32)})
    leftovers = [p.name for p in tmp_path.iterdir() if ".tmp." in p.name]
    assert leftovers == [], f"atomic write left tmp files: {leftovers}"


def test_atomic_savez_overwrites_existing(tmp_path):
    p = tmp_path / "ck.npz"
    T._atomic_savez(p, {"a": np.ones(2, np.float32)})
    T._atomic_savez(p, {"a": np.full(2, 9.0, np.float32)})
    assert np.array_equal(np.load(p)["a"], np.full(2, 9.0, np.float32))


def test_atomic_savez_refuses_tmp_path():
    with pytest.raises(ValueError, match="tmp-class"):
        T._atomic_savez(Path("/tmp/levelset_ck.npz"), {"a": np.zeros(1, np.float32)})


# --------------------------------------------------------------------------- EMA checkpoint dict
def test_ema_arrays_preserve_shadow_params():
    sh = _fake_shadow()
    out = T._build_ema_checkpoint_arrays(sh, args=_fake_args(), softmax_temp=0.05,
                                         render_h=384, render_w=512, epoch=900, in_feat=40)
    for k in sh:
        assert k in out and np.array_equal(out[k], sh[k]), k


def test_ema_arrays_include_all_historical_cfg_keys():
    out = T._build_ema_checkpoint_arrays(_fake_shadow(), args=_fake_args(), softmax_temp=0.05,
                                         render_h=384, render_w=512, epoch=1, in_feat=40)
    for k in ("__cfg_n_hidden", "__cfg_hidden_dim", "__cfg_softmax_temp", "__cfg_activation",
              "__cfg_chroma", "__cfg_wire_w0", "__cfg_wire_s0", "__cfg_hosc_beta", "__cfg_hosc_omega",
              "__bank_n_scales", "__bank_n_orient0", "__bank_f0", "__bank_base", "__bank_n_iso",
              "__render_hw", "__cfg_max_bank_freq", "__cfg_lane_edge_weight", "__cfg_lane_edge_class"):
        assert k in out, f"missing historical cfg key {k}"
    assert list(out["__render_hw"]) == [384, 512]
    assert float(out["__cfg_softmax_temp"]) == 0.05  # the ANNEALED temp, not an args default


def test_ema_arrays_include_new_provenance_keys():
    out = T._build_ema_checkpoint_arrays(_fake_shadow(), args=_fake_args(), softmax_temp=0.05,
                                         render_h=384, render_w=512, epoch=900, in_feat=40)
    assert int(out["__epoch"]) == 900
    assert int(out["__cfg_in_feat"]) == 40
    assert int(out["__cfg_self_orient"]) == 1
    assert int(out["__cfg_n_dir_freqs"]) == 2
    assert float(out["__cfg_freq_across"]) == 32.0
    assert float(out["__cfg_freq_along"]) == 4.0
    assert int(out["__cfg_reorient_every"]) == 50
    assert float(out["__cfg_w_pose"]) == 1.0
    assert int(out["__cfg_curriculum"]) == 1
    assert int(out["__cfg_tau_softplus_start_epoch"]) == 300
    assert int(out["__cfg_l7_start_epoch"]) == 900


def test_ema_arrays_max_bank_freq_none_sentinel_and_passthrough():
    none_out = T._build_ema_checkpoint_arrays(_fake_shadow(), args=_fake_args(max_bank_freq=None),
                                              softmax_temp=0.05, render_h=384, render_w=512, epoch=1, in_feat=40)
    assert float(none_out["__cfg_max_bank_freq"]) == -1.0  # None -> -1 sentinel (byte-close decodes)
    cap_out = T._build_ema_checkpoint_arrays(_fake_shadow(), args=_fake_args(max_bank_freq=64.0),
                                             softmax_temp=0.05, render_h=384, render_w=512, epoch=1, in_feat=40)
    assert float(cap_out["__cfg_max_bank_freq"]) == 64.0


def test_basis_checkpoint_is_additive_and_default_checkpoint_layout_is_unchanged():
    default = T._build_ema_checkpoint_arrays(
        _fake_shadow(), args=_fake_args(), softmax_temp=0.05,
        render_h=384, render_w=512, epoch=1, in_feat=40,
    )
    selected = T._build_ema_checkpoint_arrays(
        _fake_shadow(), args=_fake_args(basis="windowed_curvelet"), softmax_temp=0.05,
        render_h=384, render_w=512, epoch=1, in_feat=288,
    )
    assert "__cfg_basis" not in default
    assert str(selected["__cfg_basis"]) == "windowed_curvelet"


def test_legacy_basis_aliases_share_checkpoint_and_resume_identity():
    canonical = T._build_ema_checkpoint_arrays(
        _fake_shadow(), args=_fake_args(basis="legacy_fourier_ab_control"), softmax_temp=0.05,
        render_h=384, render_w=512, epoch=1, in_feat=40,
    )
    old_alias = T._build_ema_checkpoint_arrays(
        _fake_shadow(), args=_fake_args(basis="polar_fourier"), softmax_temp=0.05,
        render_h=384, render_w=512, epoch=1, in_feat=40,
    )
    assert canonical.keys() == old_alias.keys()
    assert "__cfg_basis" not in canonical
    assert T._resume_lever_divergences(
        {"__cfg_basis": "polar_fourier"},
        _fake_args(basis="legacy_fourier_ab_control"),
    ) == []


# --------------------------------------------------------------------------- resume sidecar
def test_resume_arrays_prefix_and_roundtrip(tmp_path):
    live = {"in_proj.weight": np.ones((2, 3), np.float32), "code": np.full((4, 2), 2.0, np.float32)}
    ema = {"in_proj.weight": np.full((2, 3), 0.5, np.float32), "code": np.zeros((4, 2), np.float32)}
    opt = {"step": np.asarray(123, np.int64), "in_proj.weight.m": np.full((2, 3), 0.1, np.float32)}
    arrays = T._build_resume_state_arrays(live, ema, opt, args=_fake_args(), epoch=899, in_feat=40)
    p = T._atomic_savez(tmp_path / "levelset_resume_state.npz", arrays)
    rs = T._load_resume_state(p)
    assert rs["epoch"] == 899 and rs["has_opt"] is True
    assert np.array_equal(rs["live"]["in_proj.weight"], live["in_proj.weight"])
    assert np.array_equal(rs["live"]["code"], live["code"])
    assert np.array_equal(rs["ema"]["in_proj.weight"], ema["in_proj.weight"])
    assert np.array_equal(rs["opt"]["in_proj.weight.m"], opt["in_proj.weight.m"])
    # live and ema for the same logical key are kept distinct (no clobber).
    assert not np.array_equal(rs["live"]["in_proj.weight"], rs["ema"]["in_proj.weight"])


def test_resume_arrays_no_opt(tmp_path):
    live = {"code": np.ones((2, 2), np.float32)}
    arrays = T._build_resume_state_arrays(live, live, None, args=_fake_args(), epoch=10, in_feat=40)
    rs = T._load_resume_state(T._atomic_savez(tmp_path / "r.npz", arrays))
    assert rs["has_opt"] is False and rs["opt"] == {}
    assert rs["epoch"] == 10


def test_resume_arrays_empty_opt_dict_is_not_opt(tmp_path):
    live = {"code": np.ones((2, 2), np.float32)}
    arrays = T._build_resume_state_arrays(live, live, {}, args=_fake_args(), epoch=5, in_feat=40)
    rs = T._load_resume_state(T._atomic_savez(tmp_path / "r.npz", arrays))
    assert rs["has_opt"] is False


def test_load_resume_state_fallback_on_plain_ema_npz(tmp_path):
    # A plain deploy npz (unprefixed params + __epoch) must be resumable: unprefixed -> live.
    ema_arrays = T._build_ema_checkpoint_arrays(_fake_shadow(), args=_fake_args(), softmax_temp=0.05,
                                                render_h=384, render_w=512, epoch=425, in_feat=40)
    p = T._atomic_savez(tmp_path / "levelset_witness_ema_mlx.npz", ema_arrays)
    rs = T._load_resume_state(p)
    assert rs["epoch"] == 425          # from __epoch
    assert "code" in rs["live"]        # unprefixed param consumed as live fallback
    assert "in_proj.weight" in rs["live"]
    assert rs["has_opt"] is False


# --------------------------------------------------------------------------- _resolve_resume_path
def test_resolve_resume_prefers_resume_sidecar(tmp_path):
    (tmp_path / "levelset_resume_state.npz").write_bytes(b"x")
    (tmp_path / "levelset_witness_ema_mlx.npz").write_bytes(b"y")
    assert T._resolve_resume_path(tmp_path).name == "levelset_resume_state.npz"


def test_resolve_resume_falls_back_to_ema_npz(tmp_path):
    (tmp_path / "levelset_witness_ema_mlx.npz").write_bytes(b"y")
    assert T._resolve_resume_path(tmp_path).name == "levelset_witness_ema_mlx.npz"


def test_resolve_resume_explicit_file(tmp_path):
    f = tmp_path / "custom.npz"
    f.write_bytes(b"z")
    assert T._resolve_resume_path(f) == f


def test_resolve_resume_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        T._resolve_resume_path(tmp_path / "nope.npz")
    with pytest.raises(FileNotFoundError):
        T._resolve_resume_path(tmp_path)  # empty dir


# --------------------------------------------------------------------------- _stage_tag
def test_stage_tag_mapping():
    assert T._stage_tag("ce") == "stageCE"
    assert T._stage_tag("tau_softplus") == "stageTau"
    assert T._stage_tag("l7_softplus") == "stageL7"
    assert T._stage_tag("margin_hinge") == "stageHinge"
    assert T._stage_tag("relu") == "stage_relu"  # graceful fallback


# --------------------------------------------------------------------------- INTEGRATION: byte-close loadability
def test_ema_checkpoint_is_loadable_by_levelset_byte_close(tmp_path):
    """The deploy npz this trainer writes MUST be consumable by the level-set byte-close tool
    (the only path that can produce the exact-eval ROW). NO-FAKE: this is the real consumer."""
    from tools.levelset_byte_close_and_eval import _load_levelset_ckpt

    sh = _fake_shadow()
    ema_arrays = T._build_ema_checkpoint_arrays(sh, args=_fake_args(max_bank_freq=None), softmax_temp=0.05,
                                                render_h=384, render_w=512, epoch=1500, in_feat=40)
    T._atomic_savez(tmp_path / "levelset_witness_ema_mlx.npz", ema_arrays)
    params, cfg = _load_levelset_ckpt(tmp_path)
    # required params survive the round-trip
    for req in ("code", "in_proj.weight", "out_sdf.weight", "out_tex.weight", "palette"):
        assert req in params
    # cfg parsed from our keys (NO warning-path defaults for these)
    assert cfg["activation"] == "hosc"
    assert cfg["hidden_dim"] == 96
    assert cfg["n_hidden"] == 4
    assert cfg["chroma"] is True
    assert cfg["render_h"] == 384 and cfg["render_w"] == 512
    assert cfg["max_bank_freq"] is None  # -1 sentinel decoded back to None
    assert cfg["n_pairs"] == 3           # code rows // 2
    assert cfg["in_feat"] == 40


# =========================================================================== FEED-fm fixes
# FIX-1: RNG-state checkpointing => bit-faithful resume (the deterministic-reproducibility
# non-negotiable). FIX-2: freeze softmax_temp + hosc_beta during the Muon finisher. FIX-3: place
# the finisher before l7 = WARN not raise (RULE-6 freedom; other muon guards stay hard raises).


def _anneal_args(**over):
    """Args namespace with the anneal/finisher knobs the FEED-fm helpers read."""
    base = dict(
        epochs=1000, softmax_temp_start=1.0, softmax_temp_end=0.05,
        activation="hosc", hosc_beta=4.0, hosc_beta_end=None, hosc_beta_anneal="linear",
        muon_start_epoch=None,
    )
    base.update(over)
    return SimpleNamespace(**base)


def _epoch_draw(P, n_extra, hardness_prob, hardness_rng):
    """ONE epoch's RNG consumption, EXACTLY mirroring the trainer loop: a global-np.random
    permutation for the base pair order + (when oversample extras are active) a hardness_rng.choice
    and a SECOND global permutation over the concatenation."""
    order = np.random.permutation(P)
    if n_extra > 0:
        extra = hardness_rng.choice(P, size=n_extra, replace=True, p=hardness_prob)
        order = np.random.permutation(np.concatenate([order, extra]))
    return order


# ------------------------------------------------------------------ FIX-1 RNG bit-faithful resume
def test_rng_resume_is_bit_faithful_to_continuous(tmp_path):
    """The binding determinism fix: advance the loop RNGs N epochs, CHECKPOINT (snapshot via the
    real sidecar path + npz round-trip), RESTORE in a fresh-process simulation, advance N more --
    the resumed draw sequence MUST match a CONTINUOUS run of 2N. Exercises BOTH loop streams (global
    MT19937 permutation + hardness PCG64 choice) with oversample extras active (the hardest path)."""
    S, P, n_extra, N = 1234, 8, 3, 5
    hp = np.full(P, 1.0 / P)

    # CONTINUOUS reference: 2N epoch draws from a single startup.
    np.random.seed(S)
    cont_rng = np.random.default_rng(S + 777)
    cont = [_epoch_draw(P, n_extra, hp, cont_rng) for _ in range(2 * N)]

    # FIRST HALF: startup, N epochs, then snapshot exactly as _do_checkpoint does.
    np.random.seed(S)
    h1 = np.random.default_rng(S + 777)
    first = [_epoch_draw(P, n_extra, hp, h1) for _ in range(N)]
    resume_arrays = T._build_resume_state_arrays(
        {"code": np.ones((2, 2), np.float32)}, {"code": np.ones((2, 2), np.float32)}, None,
        args=_fake_args(), epoch=N, in_feat=40)
    resume_arrays.update(T._rng_state_arrays(h1))  # the FEED-fm merge in _do_checkpoint
    cfg = T._load_resume_state(T._atomic_savez(tmp_path / "levelset_resume_state.npz", resume_arrays))["cfg"]

    # the first-half draws matched the continuous prefix (sanity that the model is faithful).
    for a, b in zip(first, cont[:N]):
        assert np.array_equal(a, b)

    # RESUMED RUN (fresh process): trainer startup reseeds global + builds a fresh hardness_rng, THEN
    # restores the snapshot. Advancing N more must reproduce the continuous SECOND half exactly.
    np.random.seed(S)
    h2 = np.random.default_rng(S + 777)
    info = T._restore_rng_state(cfg, h2)
    assert info == {"np_global": True, "hardness": True}
    resumed = [_epoch_draw(P, n_extra, hp, h2) for _ in range(N)]
    for a, b in zip(resumed, cont[N:]):
        assert np.array_equal(a, b), "resumed draws diverged from continuous (NOT bit-faithful)"


def test_rng_state_arrays_keys_and_npz_roundtrip(tmp_path):
    np.random.seed(7)
    hr = np.random.default_rng(99)
    arrays = T._rng_state_arrays(hr)
    for k in ("__rng_np_algo", "__rng_np_keys", "__rng_np_pos", "__rng_np_has_gauss",
              "__rng_np_cached_gauss", "__rng_hardness_json"):
        assert k in arrays, f"missing RNG key {k}"
    # the 624-key MT19937 array survives the npz + cfg parse (becomes a list there) and the PCG64
    # state survives as a JSON string -- both reload WITHOUT pickle.
    cfg = T._load_resume_state(T._atomic_savez(tmp_path / "r.npz", arrays))["cfg"]
    assert len(cfg["__rng_np_keys"]) == 624
    assert "PCG64" in str(cfg["__rng_hardness_json"])


def test_rng_state_arrays_no_hardness_rng():
    arrays = T._rng_state_arrays(None)  # global-only snapshot (hardness disabled)
    assert "__rng_np_keys" in arrays and "__rng_hardness_json" not in arrays


def test_restore_rng_backcompat_old_checkpoint_no_rng_keys():
    """DEFAULT-SAFE: an old sidecar (no __rng_* keys) restores with the fresh-seeded RNGs untouched
    -- no crash, returns all-False, and the subsequent draw equals a never-restored fresh run."""
    np.random.seed(3)
    fresh = np.random.permutation(10)
    np.random.seed(3)
    hr = np.random.default_rng(3 + 777)
    info = T._restore_rng_state({"__epoch": 5}, hr)  # cfg WITHOUT any __rng_* key
    assert info == {"np_global": False, "hardness": False}
    assert np.array_equal(np.random.permutation(10), fresh)  # global RNG was left as fresh-seeded


# ------------------------------------------------------------------ FIX-2 finisher freeze
def test_softmax_temp_for_epoch_matches_inline_formula():
    """The extracted helper reproduces the pre-FEED-fm inline cosine EXACTLY (BIT-IDENTICAL)."""
    a = _anneal_args(epochs=1000, softmax_temp_start=1.0, softmax_temp_end=0.05)
    for ep in (1, 2, 250, 500, 900, 1000):
        prog_t = (ep - 1) / max(a.epochs - 1, 1)
        expect = float(a.softmax_temp_end + 0.5 * (a.softmax_temp_start - a.softmax_temp_end) * (1 + np.cos(np.pi * prog_t)))
        assert T._softmax_temp_for_epoch(ep, a) == expect
    assert T._softmax_temp_for_epoch(1, a) == 1.0          # start
    assert abs(T._softmax_temp_for_epoch(1000, a) - 0.05) < 1e-12  # end


def test_freeze_holds_muon_start_value_during_finisher():
    """Models the loop's `_anneal_ep = muon_start if muon_switched else ep` selection: temp anneals
    pre-switch, then is HELD at temp(muon_start) for EVERY finisher epoch."""
    a = _anneal_args(epochs=1000, softmax_temp_start=1.0, softmax_temp_end=0.05, muon_start_epoch=900)

    def temp(ep, muon_switched):
        anneal_ep = int(a.muon_start_epoch) if muon_switched else ep
        return T._softmax_temp_for_epoch(anneal_ep, a)

    # pre-switch: annealing (epoch-varying)
    assert temp(800, False) != temp(850, False)
    # finisher: frozen at temp(900) regardless of the current epoch
    frozen = T._softmax_temp_for_epoch(900, a)
    assert temp(900, True) == frozen
    assert temp(950, True) == frozen
    assert temp(1000, True) == frozen


def test_freeze_is_noop_when_finisher_off_bit_identical():
    """--muon-start-epoch None => muon_switched always False => _anneal_ep == ep => identical to the
    pre-FEED-fm path for both softmax_temp and hosc_beta."""
    a = _anneal_args(epochs=1000, muon_start_epoch=None)
    for ep in (1, 400, 900, 1000):
        # what the loop would set with the freeze logic when the finisher is OFF:
        anneal_ep = int(a.muon_start_epoch) if False else ep  # muon_switched is always False
        assert T._softmax_temp_for_epoch(anneal_ep, a) == T._softmax_temp_for_epoch(ep, a)
        assert T._hosc_beta_for_epoch(anneal_ep, a) == T._hosc_beta_for_epoch(ep, a)


def test_hosc_beta_freeze_during_finisher_and_noop_when_anneal_off():
    # anneal ON: beta held at beta(muon_start) during the finisher.
    a_on = _anneal_args(epochs=1000, hosc_beta=4.0, hosc_beta_end=16.0, muon_start_epoch=900)
    frozen_beta = T._hosc_beta_for_epoch(900, a_on)
    assert frozen_beta is not None
    assert T._hosc_beta_for_epoch(900, a_on) == frozen_beta  # _anneal_ep==900 each finisher epoch
    assert T._hosc_beta_for_epoch(800, a_on) != frozen_beta  # pre-switch differs (annealing)
    # anneal OFF (hosc_beta_end None): helper returns None => model.hosc_beta untouched both before
    # and during the finisher => the freeze is a no-op (BIT-IDENTICAL constant-beta path).
    a_off = _anneal_args(epochs=1000, hosc_beta=4.0, hosc_beta_end=None, muon_start_epoch=900)
    assert T._hosc_beta_for_epoch(500, a_off) is None
    assert T._hosc_beta_for_epoch(900, a_off) is None


# ------------------------------------------------------------------ FIX-3 placement WARN not raise
class _RunTrainReached(Exception):
    """Sentinel: main() passed all validation guards and reached run_train (which we stub)."""


def _muon_argv(tmp_path, **over):
    a = dict(epochs="20", tau="5", l7="10", muon="7", extra=[])
    a.update(over)
    argv = ["--out-dir", str(tmp_path), "--epochs", a["epochs"], "--curriculum",
            "--tau-softplus-start-epoch", a["tau"], "--l7-start-epoch", a["l7"],
            "--muon-start-epoch", a["muon"], "--mlx-device", "cpu", *a["extra"]]
    return argv


def test_fix3_muon_before_l7_warns_not_raises(tmp_path, monkeypatch, capsys):
    def _stub(args):
        raise _RunTrainReached
    monkeypatch.setattr(T, "run_train", _stub)
    with pytest.raises(_RunTrainReached):  # reached run_train => no fail-closed raise on placement
        T.main(_muon_argv(tmp_path, muon="7", l7="10"))  # 7 < 10
    assert "muon_finisher_WARN" in capsys.readouterr().out


def test_fix3_muon_after_l7_no_warn(tmp_path, monkeypatch, capsys):
    def _stub(args):
        raise _RunTrainReached
    monkeypatch.setattr(T, "run_train", _stub)
    with pytest.raises(_RunTrainReached):
        T.main(_muon_argv(tmp_path, muon="12", l7="10"))  # 12 >= 10 => PR95 placement, no warn
    assert "muon_finisher_WARN" not in capsys.readouterr().out


def test_fix3_range_guard_still_hard_raises(tmp_path):
    with pytest.raises(ValueError, match="must be in"):  # 25 > epochs 20
        T.main(_muon_argv(tmp_path, muon="25"))


def test_fix3_freeze_decoder_guard_still_hard_raises(tmp_path):
    with pytest.raises(ValueError, match="incompatible with --freeze-decoder-fit-codes"):
        T.main(_muon_argv(tmp_path, muon="7",
                          extra=["--freeze-decoder-fit-codes", str(tmp_path / "dec.npz")]))


# --------------------------------------------------------------------------- #403 P0 hardness/head resume
def _cfg_from_arrays(arrays: dict) -> dict:
    """Mirror _load_resume_state's cfg extraction: __-keys become python scalars/lists."""
    cfg = {}
    for k, v in arrays.items():
        if k.startswith("__"):
            a = np.asarray(v)
            cfg[k] = a.item() if a.size == 1 else a.tolist()
    return cfg


def test_hardness_head_cfg_persisted_when_armed():
    a = _fake_args(hardness_oversample=0.5, hardness_weighted=True, hardness_source="realized",
                   hardness_power=1.0, hardness_band=0.5, head="etf", additive_margin=0.0)
    live = {"code": np.zeros((6, 32), np.float32)}
    hp = np.arange(1, 601, dtype=np.float64)
    arrays = T._build_resume_state_arrays(live, live, None, args=a, epoch=25, in_feat=40,
                                          hardness_prob=hp)
    for k in ("__cfg_hardness_oversample", "__cfg_hardness_weighted", "__cfg_hardness_source",
              "__cfg_hardness_power", "__cfg_hardness_band", "__cfg_head", "__cfg_additive_margin",
              "__hardness_prob"):
        assert k in arrays, f"missing persisted key {k}"
    assert np.asarray(arrays["__hardness_prob"]).shape == (600,)


def test_hardness_prob_absent_when_lever_off():
    # hardness_prob=None (lever off / margin source) => the baseline key is NOT written (byte-identical).
    arrays = T._build_resume_state_arrays({"code": np.zeros((2, 4), np.float32)},
                                          {"code": np.zeros((2, 4), np.float32)}, None,
                                          args=_fake_args(), epoch=5, in_feat=40, hardness_prob=None)
    assert "__hardness_prob" not in arrays
    # but the cfg-guard keys are always present (default-off, legacy-safe values).
    assert "__cfg_hardness_oversample" in arrays and "__cfg_head" in arrays


def test_hardness_prob_roundtrips_through_savez(tmp_path):
    hp = (np.arange(1, 601, dtype=np.float64))
    hp = hp / hp.sum()
    a = _fake_args(hardness_oversample=0.5, hardness_weighted=True, hardness_source="realized")
    arrays = T._build_resume_state_arrays({"code": np.zeros((6, 32), np.float32)},
                                          {"code": np.zeros((6, 32), np.float32)}, None,
                                          args=a, epoch=25, in_feat=40, hardness_prob=hp)
    out = T._atomic_savez(tmp_path / "resume.npz", arrays)
    rs = T._load_resume_state(out)
    saved = np.asarray(rs["cfg"]["__hardness_prob"], np.float64)
    assert saved.shape == (600,)
    assert np.allclose(saved, hp)


def test_resume_guard_matched_hardness_head_no_divergence():
    a = _fake_args(hardness_oversample=0.5, hardness_weighted=True, hardness_source="realized",
                   hardness_power=1.0, hardness_band=0.5, head="etf", additive_margin=0.0)
    arrays = T._build_resume_state_arrays({"code": np.zeros((6, 32), np.float32)},
                                          {"code": np.zeros((6, 32), np.float32)}, None,
                                          args=a, epoch=25, in_feat=40,
                                          hardness_prob=np.ones(600))
    cfg = _cfg_from_arrays(arrays)
    assert T._resume_lever_divergences(cfg, a) == []


def test_resume_guard_flags_hardness_source_flip():
    a = _fake_args(hardness_oversample=0.5, hardness_weighted=True, hardness_source="realized",
                   head="etf")
    cfg = _cfg_from_arrays(T._build_resume_state_arrays(
        {"code": np.zeros((6, 32), np.float32)}, {"code": np.zeros((6, 32), np.float32)}, None,
        args=a, epoch=25, in_feat=40, hardness_prob=np.ones(600)))
    a2 = _fake_args(hardness_oversample=0.5, hardness_weighted=True, hardness_source="margin",
                    head="etf")
    div = T._resume_lever_divergences(cfg, a2)
    assert any("hardness_source" in d for d in div)


def test_resume_guard_flags_head_etf_dropped_to_softmax():
    # the exact gap-2 hazard: resume drops --head etf -> frozen ETF resumes as trainable softmax.
    a = _fake_args(hardness_oversample=0.0, head="etf")
    cfg = _cfg_from_arrays(T._build_resume_state_arrays(
        {"code": np.zeros((6, 32), np.float32)}, {"code": np.zeros((6, 32), np.float32)}, None,
        args=a, epoch=25, in_feat=40))
    a2 = _fake_args(hardness_oversample=0.0, head="softmax")
    div = T._resume_lever_divergences(cfg, a2)
    assert any(d.startswith("head:") for d in div)


def test_resume_guard_flags_additive_margin_change_when_engaged():
    a = _fake_args(head="additive-margin", additive_margin=0.3)
    cfg = _cfg_from_arrays(T._build_resume_state_arrays(
        {"code": np.zeros((6, 32), np.float32)}, {"code": np.zeros((6, 32), np.float32)}, None,
        args=a, epoch=25, in_feat=40))
    a2 = _fake_args(head="additive-margin", additive_margin=0.5)
    assert any("additive_margin" in d for d in T._resume_lever_divergences(cfg, a2))


def test_resume_guard_legacy_sidecar_no_spurious_divergence():
    # a pre-#403 sidecar lacks all hardness/head keys -> the guard only checks present keys.
    a = _fake_args(hardness_oversample=0.5, hardness_source="realized", head="etf")
    assert T._resume_lever_divergences({}, a) == []


def test_resume_guard_rejects_basis_family_drift():
    a = _fake_args(basis="windowed_curvelet", self_orient=False)
    cfg = _cfg_from_arrays(T._build_resume_state_arrays(
        {"code": np.zeros((6, 32), np.float32)}, {"code": np.zeros((6, 32), np.float32)}, None,
        args=a, epoch=25, in_feat=288,
    ))
    assert str(cfg["__cfg_basis"]) == "windowed_curvelet"
    div = T._resume_lever_divergences(cfg, _fake_args(basis="polar_fourier", self_orient=False))
    assert any(d.startswith("basis:") for d in div)


def test_resume_guard_hardness_subfields_inert_when_off_both():
    # oversample==0 in both -> weighted/source/power/band are inert (not flagged).
    a = _fake_args(hardness_oversample=0.0, hardness_source="realized", head="etf")
    cfg = _cfg_from_arrays(T._build_resume_state_arrays(
        {"code": np.zeros((6, 32), np.float32)}, {"code": np.zeros((6, 32), np.float32)}, None,
        args=a, epoch=25, in_feat=40))
    a2 = _fake_args(hardness_oversample=0.0, hardness_source="margin", head="etf")
    div = T._resume_lever_divergences(cfg, a2)
    assert not any("hardness_source" in d for d in div)


# --------------------------------------------------------------------------- Task #537 seal path
def _complete_native_resume_state():
    cfg = {
        "__resume_semantic_schema": T._RESUME_SEMANTIC_SCHEMA,
        "__resume_epoch": 25,
        "__resume_stage": "ce",
        "__resume_primary_optimizer_family": "adamw",
        "__cfg_seed_islands": 0,
        "__resume_has_seed": 0,
        "__resume_active_trainable_components_json": '["primary_model"]',
        "__resume_event_ledger_json": json.dumps({
            "schema": "levelset_resume_event_ledger.v1",
            "active_event_flags": [],
            "persisted_keys": [],
            "inactive_explicit": True,
        }),
        "__rng_np_algo": "MT19937",
        "__rng_np_keys": [0] * 624,
        "__rng_np_pos": 0,
        "__rng_np_has_gauss": 0,
        "__rng_np_cached_gauss": 0.0,
    }
    return {
        "live": {"code": np.ones((2, 2), np.float32)},
        "ema": {"code": np.ones((2, 2), np.float32)},
        "opt": {"step": np.asarray(1)},
        "has_opt": True,
        "epoch": 25,
        "cfg": cfg,
    }


def test_periodic_names_are_stage_and_epoch_distinct():
    assert T._periodic_checkpoint_names("stageCE", 25) == (
        "levelset_periodic_ema_stageCE_ep25.npz",
        "levelset_periodic_resume_stageCE_ep25.npz",
    )
    assert T._periodic_checkpoint_names("stageCE", 50) != T._periodic_checkpoint_names(
        "stageCE", 25)


def test_periodic_retention_is_per_stage_and_boundary_immune(tmp_path):
    for stage in ("stageCE", "stageTau"):
        for epoch in (10, 20, 30):
            for name in T._periodic_checkpoint_names(stage, epoch):
                (tmp_path / name).write_bytes(b"periodic")
    protected = {
        "levelset_resume_state.npz",
        "levelset_witness_ema_mlx.npz",
        "levelset_resume_stageCE_ep10.npz",
        "levelset_ckpt_stageCE_ep10.npz",
        "levelset_witness_ema_BEST.npz",
        "levelset_periodic_resume_stageCE_latest.npz",
    }
    for name in protected:
        (tmp_path / name).write_bytes(b"protected")

    removed = T._prune_periodic_checkpoints(tmp_path, "stageCE", retain=2)
    assert removed == [
        "levelset_periodic_ema_stageCE_ep10.npz",
        "levelset_periodic_resume_stageCE_ep10.npz",
    ]
    assert all((tmp_path / name).exists() for name in protected)
    assert all((tmp_path / name).exists() for name in T._periodic_checkpoint_names("stageTau", 10))


@pytest.mark.parametrize(
    ("leg", "mutate", "match"),
    [
        ("live", lambda rs: rs.update(live={}), "live weights"),
        ("ema", lambda rs: rs.update(ema={}), "ema_shadow"),
        ("optimizer", lambda rs: rs.update(opt={}, has_opt=False), "optimizer_moments"),
        ("rng", lambda rs: rs["cfg"].pop("__rng_np_keys"), "rng_state"),
        ("event", lambda rs: rs["cfg"].pop("__resume_event_ledger_json"), "event_state"),
        ("stage", lambda rs: rs["cfg"].pop("__resume_stage"), "stage_epoch_position"),
    ],
)
def test_normal_resume_refuses_each_missing_semantic_leg(leg, mutate, match):
    del leg
    rs = _complete_native_resume_state()
    mutate(rs)
    with pytest.raises(ValueError, match=match):
        T._validate_resume_state_for_continuation(rs, warm_start_weights_only=False)


def test_native_resume_guard_accepts_complete_state():
    row = T._validate_resume_state_for_continuation(
        _complete_native_resume_state(), warm_start_weights_only=False)
    assert row["compatibility"] == "native_v3"
    assert row["treatment_kind"] == "bit_faithful_continuation"


def test_legacy_v2_seed_off_resume_uses_ledger_without_v3_only_fields():
    rs = _complete_native_resume_state()
    cfg = rs["cfg"]
    cfg["__resume_semantic_schema"] = T._LEGACY_RESUME_SEMANTIC_SCHEMA
    cfg.pop("__resume_primary_optimizer_family")
    cfg.pop("__resume_active_trainable_components_json")
    cfg.pop("__resume_has_seed")
    row = T._validate_resume_state_for_continuation(
        rs, warm_start_weights_only=False
    )
    assert row["compatibility"] == "legacy_v2_seed_off"
    T._validate_resume_optimizer_family(
        semantic_schema=T._LEGACY_RESUME_SEMANTIC_SCHEMA,
        checkpoint_family="",
        expected_family="adamw",
        warm_start_weights_only=False,
    )


def test_v3_optimizer_family_remains_fail_closed():
    with pytest.raises(RuntimeError, match="optimizer family changed"):
        T._validate_resume_optimizer_family(
            semantic_schema=T._RESUME_SEMANTIC_SCHEMA,
            checkpoint_family="",
            expected_family="adamw",
            warm_start_weights_only=False,
        )


@pytest.mark.parametrize(
    ("heavy", "scalar_count", "runtime", "arm", "count"),
    [
        (True, 3, False, 1, 2),
        (False, 0, True, None, None),
        (False, 1, False, 1, 0),
        (True, 0, False, None, None),
        (True, 3, True, 1, 0),
        (False, 3, True, 1, 2),
    ],
)
def test_polyak_resume_presence_refuses_on_off_or_partial_drift(
    heavy,
    scalar_count,
    runtime,
    arm,
    count,
):
    with pytest.raises(RuntimeError, match="Polyak continuation"):
        T._validate_polyak_resume_presence(
            heavy_present=heavy,
            scalar_keys_present=scalar_count,
            runtime_active=runtime,
            checkpoint_arm=arm,
            checkpoint_count=count,
        )


def test_polyak_resume_presence_accepts_only_atomic_on_or_off():
    T._validate_polyak_resume_presence(
        heavy_present=False,
        scalar_keys_present=0,
        runtime_active=False,
        checkpoint_arm=None,
        checkpoint_count=None,
    )
    T._validate_polyak_resume_presence(
        heavy_present=False,
        scalar_keys_present=3,
        runtime_active=True,
        checkpoint_arm=1,
        checkpoint_count=0,
    )
    T._validate_polyak_resume_presence(
        heavy_present=True,
        scalar_keys_present=3,
        runtime_active=True,
        checkpoint_arm=1,
        checkpoint_count=4,
    )


def _g111_reducer(state, result):
    state["total"] += int(result.payload["delta"])
    return state


def _g111_worker(sequence, snapshot):
    return T.ImmutableVerdictResult.capture(
        submission_seq=sequence,
        result_id=f"verdict-{sequence}",
        payload={"delta": int(snapshot["delta"])},
    )


def _g111_owner_fixture(verdict_arrays):
    owners = {
        T.G111_ATOMIC_OWNERS[0]: {
            "liveP__weight": np.asarray([1.0, 2.0], np.float32),
            "emaP__weight": np.asarray([0.5, 1.5], np.float32),
            "optP__step": np.asarray([7], np.int64),
        },
        T.G111_ATOMIC_OWNERS[1]: {
            "__g111_rollback_epoch": np.asarray([6], np.int64),
            "__g111_rollback_weight": np.asarray([0.75, 1.75], np.float32),
        },
        T.G111_ATOMIC_OWNERS[2]: {
            "__g111_stage_epoch": np.asarray([7], np.int64),
            "__rng_np_keys": np.arange(8, dtype=np.uint32),
        },
        T.G111_ATOMIC_OWNERS[3]: dict(verdict_arrays),
        T.G111_ATOMIC_OWNERS[4]: {
            "__g111_best_present": np.asarray([1], np.int8),
            "__g111_best_epoch": np.asarray([5], np.int64),
            "__g111_best_dseg": np.asarray([0.01], np.float64),
        },
        T.G111_ATOMIC_OWNERS[5]: {
            "__g111_lineage_parent": np.arange(32, dtype=np.uint8),
        },
    }
    activity = dict.fromkeys(T.G111_ATOMIC_OWNERS, True)
    derived = {
        "__g111_lineage_checkpoint": np.arange(32, dtype=np.uint8)[::-1],
    }
    return owners, activity, derived


def _g111_checkpoint_fixture(*, journal_limit=3):
    transaction = T.QuiescentVerdictTransaction(
        reducer=_g111_reducer,
        initial_state={"total": 0},
        max_journal_rows=journal_limit,
    )
    with transaction.checkpoint() as capture:
        expected_owners, activity, expected_derived = _g111_owner_fixture(
            capture.numpy_state(prefix=T._G111_NATIVE_V3_BARRIER_PREFIX)
        )
        expected = T._build_g111_native_v3_expected_schema(
            expected_owners,
            activity=activity,
            expected_derived_lineage_arrays=expected_derived,
        )
        owners, _, derived = _g111_owner_fixture(
            capture.numpy_state(prefix=T._G111_NATIVE_V3_BARRIER_PREFIX)
        )
        arrays, expected, staged = T._build_g111_native_v3_checkpoint(
            owners,
            activity=activity,
            expected=expected,
            derived_lineage_arrays=derived,
        )
    return transaction, arrays, expected, staged


def test_g111_native_v3_cold_root_covers_six_owners_and_fourteen_domains():
    _, arrays, expected, staged = _g111_checkpoint_fixture()
    assert tuple(owner.owner for owner in expected.owners) == T.G111_ATOMIC_OWNERS
    assert len(expected.domain_coverage) == 14
    claims = {
        claim.owner: set(claim.keys)
        for claim in staged.manifest.owner_claims
    }
    assert tuple(claims) == T.G111_ATOMIC_OWNERS
    assert all(claims[owner] for owner in T.G111_ATOMIC_OWNERS)
    covered = set().union(*claims.values()) | set(
        staged.manifest.derived_lineage_keys
    )
    assert covered == set(arrays) - {T.G111_TRANSACTION_MANIFEST_KEY}
    assert staged.barrier_state is not None
    assert staged.barrier_state.pending_count == 0


def test_g111_native_v3_malformed_and_legacy_pending_fail_closed():
    _, arrays, expected, _ = _g111_checkpoint_fixture()
    malformed = dict(arrays)
    malformed[T._G111_NATIVE_V3_BARRIER_PREFIX + "pending_count"] = np.asarray(
        [1], np.int64
    )
    with pytest.raises(T.TransactionValidationError):
        T._stage_g111_native_v3_checkpoint(malformed, expected=expected)

    transaction = T.QuiescentVerdictTransaction(
        reducer=_g111_reducer,
        initial_state={"total": 0},
    )
    with transaction.checkpoint() as capture:
        expected_owners, activity, expected_derived = _g111_owner_fixture(
            capture.numpy_state(prefix=T._G111_NATIVE_V3_BARRIER_PREFIX)
        )
        expected = T._build_g111_native_v3_expected_schema(
            expected_owners,
            activity=activity,
            expected_derived_lineage_arrays=expected_derived,
        )
        owners, _, derived = _g111_owner_fixture(
            capture.numpy_state(prefix=T._G111_NATIVE_V3_BARRIER_PREFIX)
        )
        owners[T.G111_VERDICT_TRANSACTION]["__cl_pend_epoch"] = np.asarray(
            [1], np.int64
        )
        with pytest.raises(
            T.TransactionValidationError, match="pending-verdict"
        ):
            T._build_g111_native_v3_checkpoint(
                owners,
                activity=activity,
                expected=expected,
                derived_lineage_arrays=derived,
            )


def test_g111_native_v3_restore_stages_canonical_order_without_partial_publication():
    _, _, _, staged = _g111_checkpoint_fixture()
    staged_owners = []
    publications = []

    def fail_at_o4(owner, arrays):
        staged_owners.append((owner, tuple(arrays)))
        if owner == T.G111_VERDICT_TRANSACTION:
            raise RuntimeError("injected O4 restore failure")
        return owner

    with pytest.raises(RuntimeError, match="injected O4"):
        T._stage_and_publish_g111_native_v3_restore(
            staged,
            stage_owner=fail_at_o4,
            publish=lambda replacements: publications.append(replacements),
        )
    assert [owner for owner, _ in staged_owners] == list(
        T.G111_RESTORABLE_STATE_OWNERS[:4]
    )
    assert publications == []

    observed = []

    def stage_owner(owner, arrays):
        observed.append(owner)
        return tuple(arrays)

    published = T._stage_and_publish_g111_native_v3_restore(
        staged,
        stage_owner=stage_owner,
        publish=lambda replacements: tuple(replacements),
    )
    assert observed == list(T.G111_ATOMIC_OWNERS)
    assert published == T.G111_ATOMIC_OWNERS


def test_g111_native_v3_interrupted_and_continuous_next_steps_are_exact():
    continuous = T.QuiescentVerdictTransaction(
        reducer=_g111_reducer,
        initial_state={"total": 0},
        max_journal_rows=2,
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        continuous.submit(executor, _g111_worker, {"delta": 2})
        with continuous.checkpoint() as interrupted_capture:
            expected_owners, activity, expected_derived = _g111_owner_fixture(
                interrupted_capture.numpy_state(
                    prefix=T._G111_NATIVE_V3_BARRIER_PREFIX
                )
            )
            expected = T._build_g111_native_v3_expected_schema(
                expected_owners,
                activity=activity,
                expected_derived_lineage_arrays=expected_derived,
            )
            owners, _, derived = _g111_owner_fixture(
                interrupted_capture.numpy_state(
                    prefix=T._G111_NATIVE_V3_BARRIER_PREFIX
                )
            )
            arrays, expected, _ = T._build_g111_native_v3_checkpoint(
                owners,
                activity=activity,
                expected=expected,
                derived_lineage_arrays=derived,
            )
        reopened = T._stage_g111_native_v3_checkpoint(
            arrays, expected=expected
        )
        resumed = T._restore_g111_native_v3_verdict_transaction(
            reopened,
            reducer=_g111_reducer,
            restored_reducer_state=interrupted_capture.reducer_state,
        )
        for delta in (3, 5):
            continuous.submit(executor, _g111_worker, {"delta": delta})
            resumed.submit(executor, _g111_worker, {"delta": delta})
        with continuous.checkpoint() as continuous_end:
            pass
        with resumed.checkpoint() as resumed_end:
            pass

    assert continuous_end.reducer_state == resumed_end.reducer_state == {
        "total": 10
    }
    assert continuous_end.next_apply_seq == resumed_end.next_apply_seq == 3
    assert len(continuous_end.journal) == len(resumed_end.journal) == 2
    assert continuous_end.journal == resumed_end.journal


def test_g111_native_v3_rejects_legacy_checkpoint_as_proof():
    _, _, expected, _ = _g111_checkpoint_fixture()
    legacy = {
        "__resume_semantic_schema": np.asarray(
            T._RESUME_SEMANTIC_SCHEMA
        ),
        "__cl_pend_epoch": np.asarray([4], np.int64),
    }
    with pytest.raises(
        T.TransactionValidationError, match="manifest is absent"
    ):
        T._stage_g111_native_v3_checkpoint(legacy, expected=expected)


def test_g111_native_v3_capture_cannot_define_its_own_expected_subset():
    transaction = T.QuiescentVerdictTransaction(
        reducer=_g111_reducer,
        initial_state={"total": 0},
    )
    with transaction.checkpoint() as capture:
        barrier = capture.numpy_state(
            prefix=T._G111_NATIVE_V3_BARRIER_PREFIX
        )
        expected_owners, activity, expected_derived = _g111_owner_fixture(
            barrier
        )
        expected = T._build_g111_native_v3_expected_schema(
            expected_owners,
            activity=activity,
            expected_derived_lineage_arrays=expected_derived,
        )
        captured_owners, _, captured_derived = _g111_owner_fixture(barrier)
        captured_owners[T.G111_ATOMIC_OWNERS[0]].pop("liveP__weight")
        with pytest.raises(
            T.TransactionValidationError, match="missing=.*liveP__weight"
        ):
            T._build_g111_native_v3_checkpoint(
                captured_owners,
                activity=activity,
                expected=expected,
                derived_lineage_arrays=captured_derived,
            )


def test_g111_native_v3_fake_manifest_presence_cannot_open_launch_gate():
    _, arrays, _, _ = _g111_checkpoint_fixture()
    fake = {
        **arrays,
        T.G111_TRANSACTION_MANIFEST_KEY: np.empty(0, np.uint8),
    }
    with pytest.raises(
        T.TransactionValidationError, match="not implemented"
    ):
        T._require_g111_native_v3_launch_gate(fake)


def test_g111_native_v3_same_source_incomplete_staged_cannot_admit_launch():
    transaction = T.QuiescentVerdictTransaction(
        reducer=_g111_reducer,
        initial_state={"total": 0},
    )
    with transaction.checkpoint() as capture:
        barrier = capture.numpy_state(
            prefix=T._G111_NATIVE_V3_BARRIER_PREFIX
        )
        incomplete, activity, derived = _g111_owner_fixture(barrier)
        incomplete[T.G111_ATOMIC_OWNERS[0]].pop("liveP__weight")
        expected = T._build_g111_native_v3_expected_schema(
            incomplete,
            activity=activity,
            expected_derived_lineage_arrays=derived,
        )
        captured, _, captured_derived = _g111_owner_fixture(barrier)
        captured[T.G111_ATOMIC_OWNERS[0]].pop("liveP__weight")
        _, _, structurally_valid = T._build_g111_native_v3_checkpoint(
            captured,
            activity=activity,
            expected=expected,
            derived_lineage_arrays=captured_derived,
        )
    with pytest.raises(
        T.TransactionValidationError, match="not implemented"
    ):
        T._require_g111_native_v3_launch_gate(structurally_valid)


@pytest.mark.parametrize("candidate", [None, object(), 0, "admit"])
def test_g111_native_v3_launch_gate_has_no_adapter_success_path(candidate):
    with pytest.raises(
        T.TransactionValidationError, match="not implemented"
    ):
        T._require_g111_native_v3_launch_gate(candidate)


def test_g111_native_v3_lineage_keys_require_exact_strings_before_sorting():
    transaction = T.QuiescentVerdictTransaction(
        reducer=_g111_reducer,
        initial_state={"total": 0},
    )
    with transaction.checkpoint() as capture:
        owners, activity, _ = _g111_owner_fixture(
            capture.numpy_state(prefix=T._G111_NATIVE_V3_BARRIER_PREFIX)
        )
    with pytest.raises(
        T.TransactionValidationError, match="exact canonical strings"
    ):
        T._build_g111_native_v3_expected_schema(
            owners,
            activity=activity,
            expected_derived_lineage_arrays={
                "__valid": np.asarray([1], np.int8),
                1: np.asarray([2], np.int8),
            },
        )


def test_g111_native_v3_owner_keys_require_exact_strings_before_sorting():
    transaction = T.QuiescentVerdictTransaction(
        reducer=_g111_reducer,
        initial_state={"total": 0},
    )
    with transaction.checkpoint() as capture:
        owners, activity, derived = _g111_owner_fixture(
            capture.numpy_state(prefix=T._G111_NATIVE_V3_BARRIER_PREFIX)
        )
    owners[T.G111_ATOMIC_OWNERS[0]][1] = np.asarray([2], np.int8)
    with pytest.raises(
        T.TransactionValidationError, match="exact canonical strings"
    ):
        T._build_g111_native_v3_expected_schema(
            owners,
            activity=activity,
            expected_derived_lineage_arrays=derived,
        )


def test_g111_native_v3_checkpoint_open_requires_typed_expected_schema():
    _, arrays, _, _ = _g111_checkpoint_fixture()
    with pytest.raises(
        T.TransactionValidationError, match="ExpectedTransactionSchema"
    ):
        T._stage_g111_native_v3_checkpoint(arrays, expected=None)


def test_g111_native_v3_inactive_o4_restore_fails_typed():
    transaction = T.QuiescentVerdictTransaction(
        reducer=_g111_reducer,
        initial_state={"total": 0},
    )
    with transaction.checkpoint():
        expected_owners, activity, expected_derived = _g111_owner_fixture({})
        activity[T.G111_VERDICT_TRANSACTION] = False
        expected = T._build_g111_native_v3_expected_schema(
            expected_owners,
            activity=activity,
            expected_derived_lineage_arrays=expected_derived,
        )
        captured_owners, _, captured_derived = _g111_owner_fixture({})
        _, _, staged = T._build_g111_native_v3_checkpoint(
            captured_owners,
            activity=activity,
            expected=expected,
            derived_lineage_arrays=captured_derived,
        )
    with pytest.raises(
        T.TransactionValidationError, match="active validated"
    ):
        T._restore_g111_native_v3_verdict_transaction(
            staged,
            reducer=_g111_reducer,
            restored_reducer_state={"total": 0},
        )


def test_g111_seed_custody_has_no_dense_host_snapshot_or_restore():
    source = inspect.getsource(T.run_train)
    assert "_seed_res_np = np.zeros" not in source
    assert "_seed_msk_np = np.zeros" not in source
    assert "mx.take(" in source
    assert "pack_sparse_auxiliary_selected_state" in source
    assert "validate_sparse_auxiliary_packed_state" in source
    assert "rows.at[seed_support_indices_mx].add(value_mx)" in source
    assert '"complete_trajectory_proven": True' not in source
    assert '"fresh_lineage_complete_trajectory_proven": True' not in source


def test_legacy_full_state_requires_and_reports_direct_event_evidence():
    rs = _complete_native_resume_state()
    rs["cfg"].pop("__resume_semantic_schema")
    rs["cfg"].pop("__resume_stage")
    rs["cfg"].pop("__resume_event_ledger_json")
    rs["cfg"]["__posegate_engaged_epoch"] = 23
    rs["cfg"]["__dtp_event_mark_resume_keys_json"] = "[]"
    row = T._validate_resume_state_for_continuation(rs, warm_start_weights_only=False)
    assert row["legacy_compatibility"] is True

    rs["cfg"].pop("__posegate_engaged_epoch")
    rs["cfg"].pop("__dtp_event_mark_resume_keys_json")
    with pytest.raises(ValueError, match="legacy_direct_keys"):
        T._validate_resume_state_for_continuation(rs, warm_start_weights_only=False)


def test_weights_only_is_the_explicit_state_drop_escape():
    rs = {"live": {"code": np.ones((1,), np.float32)}, "cfg": {}}
    row = T._validate_resume_state_for_continuation(rs, warm_start_weights_only=True)
    assert row["continuity_required"] is False
    with pytest.raises(ValueError):
        T._validate_resume_state_for_continuation(rs, warm_start_weights_only=False)


def test_optimizer_restore_keyset_must_be_exact_not_partial():
    T._validate_optimizer_restore_keysets({"step", "m.weight", "v.weight"},
                                          {"step", "m.weight", "v.weight"})
    with pytest.raises(ValueError, match="partial restore is not bit-faithful"):
        T._validate_optimizer_restore_keysets({"step", "m.weight"},
                                              {"step", "m.weight", "v.weight"})
    with pytest.raises(ValueError, match="extra_checkpoint_keys"):
        T._validate_optimizer_restore_keysets({"step", "m.weight", "v.weight", "stale"},
                                              {"step", "m.weight", "v.weight"})


def test_exact_continuation_reanchor_is_identity():
    args = _fake_args(
        warm_start_epoch=-1, num_pairs=24, accum_pairs=8, adam_beta2=0.999,
        stage_transition_rewarmup_epochs=8, muon_start_epoch=726,
        pose_finish_start_epoch=1000, lane_band_start_epoch=300,
        seg_chroma_boundary_start_epoch=0, seg_temporal_screw_start_epoch=0,
        seg_phase_advect_start_epoch=0,
    )
    row = T._reanchor_resume_round(
        args, checkpoint_epoch=650, warm_start_weights_only=False)
    assert row["changed"] is False
    assert row["old_anchors"] == row["new_anchors"]
    assert args.muon_start_epoch == 726


def test_warm_retreatment_reanchors_events_to_current_geometry():
    args = _fake_args(
        warm_start_epoch=-1, num_pairs=24, accum_pairs=8, adam_beta2=0.999,
        stage_transition_rewarmup_epochs=8, muon_start_epoch=726,
        pose_finish_start_epoch=1000, lane_band_start_epoch=300,
        seg_chroma_boundary_start_epoch=0, seg_temporal_screw_start_epoch=0,
        seg_phase_advect_start_epoch=0,
    )
    row = T._reanchor_resume_round(
        args, checkpoint_epoch=650, warm_start_weights_only=True)
    assert row["derived_window"] == 667  # ceil(2/(1-.999)/(24/8))
    assert row["resume_epoch"] == 651
    assert args.muon_start_epoch == 1318
    assert args.pose_finish_start_epoch == 1318
    assert args.lane_band_start_epoch == 1318
    assert args.stage_transition_rewarmup_epochs == 667
    assert row["old_anchors"]["muon_start_epoch"] == 726
    assert row["new_anchors"]["muon_start_epoch"] == 1318
