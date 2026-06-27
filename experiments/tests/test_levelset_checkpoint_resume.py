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
