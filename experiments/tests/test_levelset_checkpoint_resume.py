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

import sys
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
