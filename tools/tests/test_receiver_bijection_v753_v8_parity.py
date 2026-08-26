# SPDX-License-Identifier: MIT
"""#417 fix-half PARITY GATE: the byte-close receiver numpy forward CONSUMES the v7.5.3 tex_trunk /
out_tex_h + v8 decoupled_head weight groups, and its output MATCHES the trainer MLX forward.

This is the MEASURED proof that the fix is REAL, not FAKE (NO-FAKE #8 + the numpy-fp32-is-the-
bit-identical-authority non-negotiable): each group's receiver mirror is compared OP-FOR-OP against
the ACTUAL trainer MLX submodule (``make_texture_trunk_mlx`` / ``make_decoupled_field_head_mlx``) or
the trainer's exact head algebra, on the SAME params + coords + code. Parity is float32-level (the
numpy reference runs fp64 then the receiver renders in its fp64 default; the MLX module is fp32 --
agreement is relmax ~1e-5..1e-4, NOT bit-exact, per the module docstrings). A group whose receiver
mirror did NOT match would be an inert lever -> a fake scored A/B; this test refuses that.

Also asserts (a) a SHARED-HEAD witness forward is BYTE-IDENTICAL (the new branches skipped), and
(b) the receiver-consumption bijection gate now reports ZERO orphans for the three groups.
"""
from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[2]
_TOOLS = _ROOT / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from tac.boundary_math.decoupled_field import (  # noqa: E402
    DecoupledFieldSpec,
    make_decoupled_field_head_mlx,
)
from tac.boundary_math.texture_trunk import (  # noqa: E402
    TextureBandSpec,
    make_texture_trunk_mlx,
)


def _load_tool_module():
    spec = importlib.util.spec_from_file_location("lbce_417", _TOOLS / "levelset_byte_close_and_eval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _receiver_ns():
    """exec the shipped ``_INFLATE_PY`` source string in an isolated namespace (NOT __main__ -> main()
    is not called) so we test the EXACT receiver functions that ship in inflate.py."""
    src = _load_tool_module()._INFLATE_PY
    ns: dict = {"__name__": "_recv_inflate_test_417"}
    exec(compile(src, "inflate.py", "exec"), ns)  # noqa: S102 (executing our own shipped source)
    return ns


_NS = _receiver_ns()
# ISOLATED group-forward parity (receiver mirror vs the exact MLX submodule): tight float32 level.
_RELTOL_ISO = 2e-3
# END-TO-END compose parity (through softmax -> palette -> sigmoid*255): the sigmoid on a 255 scale
# AMPLIFIES the fp64(receiver)-vs-fp32(MLX) last-bit divergence of argmax-boundary-adjacent logits, so
# ~a few 1e-3 relmax = <~2 uint8/255 is EXPECTED + benign for argmax-d_seg / MSE-d_pose. Documented,
# measured, printed (NOT a bit-exact claim -- the numpy-fp32-authority parity is float32-level).
_RELTOL_COMPOSE = 8e-3


def _relmax(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    denom = float(np.max(np.abs(b))) + 1e-12
    return float(np.max(np.abs(a - b)) / denom)


# --------------------------------------------------------------------------- #
# Group 1: tex_trunk (#395) -- receiver _tex_trunk_forward vs the MLX module.  #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("annulus_power", [0.0, 1.5])
def test_tex_trunk_parity(annulus_power: float) -> None:
    rh, rw, K = 6, 8, 5
    spec = TextureBandSpec(band_hi=8.0)
    trunk = make_texture_trunk_mlx(rh, rw, spec, n_classes=K, annulus_power=annulus_power,
                                   coeff_scale=0.05, seed=3)
    P = rh * rw
    rng = np.random.default_rng(11)
    logits = rng.standard_normal((P, K)).astype(np.float32)
    soft_np = np.exp(logits - logits.max(-1, keepdims=True))
    soft_np = (soft_np / soft_np.sum(-1, keepdims=True)).astype(np.float32)

    mlx_tex = np.asarray(trunk(mx.array(soft_np)))  # the trainer forward (the actual submodule)

    w_tex = np.asarray(trunk.w_tex); bias = np.asarray(trunk.bias)
    bank = _NS["_tex_trunk_bank"](rh, rw)
    # the bank the receiver regenerates FREE must equal the trainer's stored (excluded) bank_B.
    assert _relmax(bank, np.asarray(trunk.bank_B)) < 1e-6, "regenerated Gabor bank != trainer bank_B"
    recv_tex = _NS["_tex_trunk_forward"](bank, w_tex, bias, soft_np, K, annulus_power)

    rel = _relmax(recv_tex, mlx_tex)
    print(f"[tex_trunk parity ap={annulus_power}] relmax={rel:.3e}")
    assert rel < _RELTOL_ISO, f"tex_trunk receiver != MLX (relmax={rel:.3e})"


# --------------------------------------------------------------------------- #
# Group 2: decoupled_head (v8 B1) -- receiver _decoupled_phi vs the MLX module.#
# --------------------------------------------------------------------------- #
def test_decoupled_head_parity() -> None:
    in_feat, mod_dim, K, H, L = 12, 7, 5, 8, 2
    spec = DecoupledFieldSpec(in_feat=in_feat, mod_dim=mod_dim, n_classes=K,
                              field_hidden=H, field_layers=L, activation="relu")
    head = make_decoupled_field_head_mlx(spec, seed=5, scale=0.2)
    P = 40
    rng = np.random.default_rng(7)
    feats = rng.standard_normal((P, in_feat)).astype(np.float32)
    code = rng.standard_normal((mod_dim,)).astype(np.float32)

    mlx_phi = np.asarray(head.phi_single(mx.array(feats), mx.array(code)))  # (P, K) trainer forward

    Pdict = {
        "decoupled_head.w_in": np.asarray(head.w_in), "decoupled_head.b_in": np.asarray(head.b_in),
        "decoupled_head.w_film": np.asarray(head.w_film), "decoupled_head.w_hid": np.asarray(head.w_hid),
        "decoupled_head.b_hid": np.asarray(head.b_hid), "decoupled_head.w_out": np.asarray(head.w_out),
        "decoupled_head.b_out": np.asarray(head.b_out),
    }
    recv_phi = _NS["_decoupled_phi"](Pdict, feats, code)

    rel = _relmax(recv_phi, mlx_phi)
    print(f"[decoupled_head parity] relmax={rel:.3e} shape={recv_phi.shape}")
    assert recv_phi.shape == (P, K)
    assert rel < _RELTOL_ISO, f"decoupled_head receiver != MLX (relmax={rel:.3e})"


# --------------------------------------------------------------------------- #
# End-to-end: _outputs_from_h0 rgb == an MLX compose reference, per group.     #
# Proves the ACTUAL shipped receiver head code (out_tex_h widen + trunk add +  #
# decoupled phi) matches the trainer _compose_rgb, and byte-identity for the   #
# shared head.                                                                 #
# --------------------------------------------------------------------------- #
def _mk_base_params(K: int, hidden: int, mod: int, n_hidden: int, in_feat: int, seed: int):
    rng = np.random.default_rng(seed)
    P: dict[str, np.ndarray] = {
        "film.weight": rng.standard_normal((n_hidden * 2 * hidden, mod)).astype(np.float32) * 0.1,
        "film.bias": rng.standard_normal((n_hidden * 2 * hidden,)).astype(np.float32) * 0.1,
        "out_sdf.weight": rng.standard_normal((K, hidden)).astype(np.float32) * 0.3,
        "out_sdf.bias": rng.standard_normal((K,)).astype(np.float32) * 0.1,
        "out_tex.weight": rng.standard_normal((3, hidden)).astype(np.float32) * 0.2,
        "out_tex.bias": rng.standard_normal((3,)).astype(np.float32) * 0.1,
        "palette": rng.standard_normal((K, 3)).astype(np.float32),
    }
    for li in range(n_hidden):
        P[f"hidden.{li}.weight"] = rng.standard_normal((hidden, hidden)).astype(np.float32) * 0.2
        P[f"hidden.{li}.bias"] = rng.standard_normal((hidden,)).astype(np.float32) * 0.1
    return P


def _mlx_compose_reference(P, h0, code, m, feats, *, out_tex_h=None, trunk=None, decoupled=None):
    """mirror of the trainer LevelSetRGBWitness._trunk + _compose_rgb (relu act, chroma on)."""
    hidden = m["hidden_dim"]; n_hidden = m["n_hidden"]
    h = mx.array(np.asarray(h0, np.float32))
    cr = mx.array(np.asarray(code, np.float32))
    film = mx.reshape(cr @ mx.array(P["film.weight"]).T + mx.array(P["film.bias"]), (n_hidden, 2, hidden))
    for li in range(n_hidden):
        scale = 1.0 + film[li, 0]; shift = film[li, 1]
        pre = (h @ mx.array(P[f"hidden.{li}.weight"]).T + mx.array(P[f"hidden.{li}.bias"])) * scale + shift
        h = nn.relu(pre)
    if decoupled is not None:
        phi = decoupled.phi_single(mx.array(np.asarray(feats, np.float32)), cr)
    else:
        phi = h @ mx.array(P["out_sdf.weight"]).T + mx.array(P["out_sdf.bias"])
    if out_tex_h is not None:
        wh, bh = out_tex_h
        tex = nn.relu(h @ mx.array(wh).T + mx.array(bh)) @ mx.array(P["out_tex.weight"]).T + mx.array(P["out_tex.bias"])
    else:
        tex = h @ mx.array(P["out_tex.weight"]).T + mx.array(P["out_tex.bias"])
    soft = mx.softmax(phi / float(m["softmax_temp"]), axis=-1)
    if trunk is not None:
        tex = tex + trunk(soft)
    rgb = mx.sigmoid(soft @ mx.array(P["palette"]) + tex) * 255.0
    return np.asarray(rgb)


def _base_m(K, hidden, n_hidden, rh, rw):
    return {"activation": "relu", "wire_w0": 20.0, "wire_s0": 10.0, "hosc_beta": 4.0, "hosc_omega": 1.0,
            "n_hidden": n_hidden, "hidden_dim": hidden, "softmax_temp": 0.05, "chroma": True,
            "render_h": rh, "render_w": rw}


def test_compose_end_to_end_all_groups() -> None:
    K, hidden, mod, n_hidden = 5, 8, 7, 2
    rh, rw = 6, 8
    Pn = rh * rw
    in_feat = 12
    P = _mk_base_params(K, hidden, mod, n_hidden, in_feat, seed=21)
    rng = np.random.default_rng(99)
    h0 = rng.standard_normal((Pn, hidden)).astype(np.float32)
    code = rng.standard_normal((mod,)).astype(np.float32)
    feats = rng.standard_normal((Pn, in_feat)).astype(np.float32)
    outfn = _NS["_outputs_from_h0"]

    # ---- shared head (baseline) ----
    m = _base_m(K, hidden, n_hidden, rh, rw)
    _phi, rgb_shared = outfn(P, h0, code, m, True, feats=feats)
    ref = _mlx_compose_reference(P, h0, code, m, feats)
    print(f"[shared-head compose] relmax={_relmax(rgb_shared, ref):.3e}")
    assert _relmax(rgb_shared, ref) < _RELTOL_COMPOSE

    # ---- + out_tex_h (widened texture head #395 A2) ----
    N = 6
    wh = rng.standard_normal((N, hidden)).astype(np.float32) * 0.2
    bh = rng.standard_normal((N,)).astype(np.float32) * 0.1
    Ph = dict(P)
    Ph["out_tex_h.weight"] = wh; Ph["out_tex_h.bias"] = bh
    Ph["out_tex.weight"] = rng.standard_normal((3, N)).astype(np.float32) * 0.2  # reshaped N->3
    _phi, rgb_h = outfn(Ph, h0, code, m, True, feats=feats)
    ref_h = _mlx_compose_reference(Ph, h0, code, m, feats, out_tex_h=(wh, bh))
    rel = _relmax(rgb_h, ref_h)
    print(f"[out_tex_h compose] relmax={rel:.3e}")
    assert rel < _RELTOL_COMPOSE
    # and it must actually CHANGE the render vs shared (positive control: a live lever, not inert)
    assert _relmax(rgb_h, rgb_shared) > 1e-2, "out_tex_h did not change the render (inert!)"

    # ---- + tex_trunk (#395) ----
    spec = TextureBandSpec(band_hi=8.0)
    trunk = make_texture_trunk_mlx(rh, rw, spec, n_classes=K, annulus_power=0.0, coeff_scale=0.1, seed=4)
    Pt = dict(P)
    Pt["tex_trunk.w_tex"] = np.asarray(trunk.w_tex); Pt["tex_trunk.bias"] = np.asarray(trunk.bias)
    mt = dict(m); mt["texture_trunk_annulus_power"] = 0.0
    _phi, rgb_t = outfn(Pt, h0, code, mt, True, feats=feats)
    ref_t = _mlx_compose_reference(P, h0, code, m, feats, trunk=trunk)
    rel = _relmax(rgb_t, ref_t)
    print(f"[tex_trunk compose] relmax={rel:.3e}")
    assert rel < _RELTOL_COMPOSE
    assert _relmax(rgb_t, rgb_shared) > 1e-3, "tex_trunk did not change the render (inert!)"

    # ---- + decoupled_head (v8 B1) ----
    dspec = DecoupledFieldSpec(in_feat=in_feat, mod_dim=mod, n_classes=K, field_hidden=8,
                               field_layers=2, activation="relu")
    head = make_decoupled_field_head_mlx(dspec, seed=6, scale=0.25)
    Pd = dict(P)
    for nm in ("w_in", "b_in", "w_film", "w_hid", "b_hid", "w_out", "b_out"):
        Pd[f"decoupled_head.{nm}"] = np.asarray(getattr(head, nm))
    _phi, rgb_d = outfn(Pd, h0, code, m, True, feats=feats)
    ref_d = _mlx_compose_reference(P, h0, code, m, feats, decoupled=head)
    rel = _relmax(rgb_d, ref_d)
    print(f"[decoupled_head compose] relmax={rel:.3e}")
    assert rel < _RELTOL_COMPOSE
    assert _relmax(rgb_d, rgb_shared) > 1e-2, "decoupled_head did not change the render (inert!)"


# --------------------------------------------------------------------------- #
# Byte-identity: a shared-head witness forward is UNCHANGED (fp64 bit-exact).  #
# --------------------------------------------------------------------------- #
def test_shared_head_forward_byte_identical() -> None:
    """The shared-head path (no tex_trunk/out_tex_h/decoupled keys, feats given AND not) must produce
    a bit-identical rgb regardless of the new feats arg -> the new branches are provably inert."""
    K, hidden, mod, n_hidden = 5, 8, 7, 3
    rh, rw = 5, 7
    Pn = rh * rw
    P = _mk_base_params(K, hidden, mod, n_hidden, 12, seed=33)
    rng = np.random.default_rng(1)
    h0 = rng.standard_normal((Pn, hidden)).astype(np.float32)
    code = rng.standard_normal((mod,)).astype(np.float32)
    feats = rng.standard_normal((Pn, 12)).astype(np.float32)
    m = _base_m(K, hidden, n_hidden, rh, rw)
    outfn = _NS["_outputs_from_h0"]
    _p0, rgb_no_feats = outfn(P, h0, code, m, True)              # feats=None (old call shape)
    _p1, rgb_feats = outfn(P, h0, code, m, True, feats=feats)    # feats supplied
    # bit-exact: the shared head ignores feats entirely (no decoupled key present).
    assert np.array_equal(rgb_no_feats, rgb_feats), "shared-head render depends on feats -> NOT inert!"
    # phi-only (want_rgb=False) path also inert to feats
    p_no, _ = outfn(P, h0, code, m, False)
    p_yes, _ = outfn(P, h0, code, m, False, feats=feats)
    assert np.array_equal(p_no, p_yes)


# --------------------------------------------------------------------------- #
# Gate: zero orphans for the three groups; still clean for the shared head.    #
# --------------------------------------------------------------------------- #
def test_gate_zero_orphans_for_new_groups() -> None:
    from levelset_receiver_bijection_gate import check_receiver_bijection

    src = _load_tool_module()._INFLATE_PY
    counted = [
        "in_proj.weight", "in_proj.bias", "film.weight", "film.bias",
        "hidden.0.weight", "hidden.0.bias", "out_sdf.weight", "out_sdf.bias",
        "out_tex.weight", "out_tex.bias", "palette",
        # the three previously-orphaned groups (#417):
        "out_tex_h.weight", "out_tex_h.bias",
        "tex_trunk.w_tex", "tex_trunk.bias",
        "decoupled_head.w_in", "decoupled_head.b_in", "decoupled_head.w_film",
        "decoupled_head.w_hid", "decoupled_head.b_hid", "decoupled_head.w_out", "decoupled_head.b_out",
    ]
    orphans, vocab = check_receiver_bijection(counted, src)
    assert not vocab.dynamic_access, "receiver has a dynamic P-access -> gate degraded to advisory"
    assert orphans == [], f"receiver still orphans counted groups: {orphans}"


def test_gate_clean_on_real_shared_head_witnesses() -> None:
    """No regression: the real shared-head witness param sets (curvelet + FiLM + out_sdf/out_tex +
    palette, and the pose_carrier/_B exclusions) still have ZERO orphans."""
    from levelset_receiver_bijection_gate import check_receiver_bijection

    src = _load_tool_module()._INFLATE_PY
    npzs = sorted(_ROOT.glob("experiments/results/**/levelset_witness_ema_mlx.npz"))[:8]
    if not npzs:
        pytest.skip("no real witness npz available")
    checked = 0
    for npz in npzs:
        z = np.load(npz, allow_pickle=False)
        params = [k for k in z.files if not k.startswith("__")]
        # mirror build_levelset_blob base_order: drop code / pose_carrier.* / B / *_B
        base_order = [k for k in params if k != "code" and not k.startswith("pose_carrier.")
                      and not (k == "B" or k.endswith("_B"))]
        orphans, vocab = check_receiver_bijection(base_order, src)
        assert not vocab.dynamic_access
        assert orphans == [], f"{npz.name}: shared-head witness now has orphans {orphans} (REGRESSION)"
        checked += 1
    assert checked >= 1


# also assert the tool base_order actually excludes _B / pose_carrier (the tool-side fix half).
def test_tool_base_order_excludes_free_bank_and_pose_carrier() -> None:
    src_text = (_TOOLS / "levelset_byte_close_and_eval.py").read_text()
    tree = ast.parse(src_text)
    # find the base_order list-comp in build_levelset_blob and confirm it filters _B + pose_carrier.
    found = any(
        isinstance(n, (ast.Assign, ast.AnnAssign))
        and n.value is not None
        and any(
            getattr(t, "id", "") == "base_order"
            for t in (n.targets if isinstance(n, ast.Assign) else [n.target])
        )
        and "endswith" in ast.dump(n.value)
        and "pose_carrier." in ast.dump(n.value)
        for n in ast.walk(tree)
    )
    assert found, "build_levelset_blob base_order must exclude _B free tables AND pose_carrier.*"
