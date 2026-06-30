# SPDX-License-Identifier: MIT
"""NO-FAKE tests for the v2 residual-only training mode + inflate residual-compose hook.

Covers (Deliverable 1 + 2 of the residual-only build):
  * residual_compose: the bulk-label-derived composition mask + the where() compose op + the
    training-bundle roundtrip (+ the /tmp-refusal guard).
  * archive_grammar: the residual-INR blob build/parse roundtrip + the 4-section byte accounting.
  * INFLATE PARITY (the NO-FAKE anchor): the self-contained inflate.py (run as a subprocess, like
    the contest) composes ``where(bulk_label_mask, INR, bulk)`` BIT-IDENTICALLY to the numpy oracle
    (residual_inflate_reference) -- WITH a real per-class warp -- and the empty-residual archive
    still produces the deterministic floor (frame0 == frame1).
  * the render seam (sibling): render_through_R_mlx(compose_fn=None) is BYTE-IDENTICAL to the
    pre-residual bare render (the baseline-path-unchanged proof); a compose_fn actually composes.
  * the surgical-lever coherence requirement: a lever (lane-thin) measurably reweights the realized
    seg loss on the COMPOSED render (not a silent no-op), so the theta* sweep runs on the hybrid.
  * the trainer's fail-closed residual-mode config guards.

means != ends: this validates the BUILD. The frontier pointer is UNMOVED 0.19110; it moves only on
a byte-closed ``upstream/evaluate.py`` row from the (HELD) residual-INR GPU run.
"""
from __future__ import annotations

import importlib.util
import struct
import subprocess
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
for _p in (REPO, REPO / "src", REPO / "experiments", REPO / "upstream"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

_HAS_MLX = importlib.util.find_spec("mlx") is not None
_HAS_TORCH = importlib.util.find_spec("torch") is not None


def _scorer_weights_present() -> bool:
    return (REPO / "upstream" / "models").exists() and _HAS_TORCH


def _load_levelset_trainer():
    name = "train_levelset_witness_realized_through_R_mlx"
    if name in sys.modules:
        return sys.modules[name]
    # the sibling must be importable first (the level-set trainer imports from it at module scope).
    sib = "train_witness_realized_through_R_mlx"
    if sib not in sys.modules:
        sspec = importlib.util.spec_from_file_location(sib, REPO / "experiments" / f"{sib}.py")
        smod = importlib.util.module_from_spec(sspec)
        sys.modules[sib] = smod
        sspec.loader.exec_module(smod)
    spec = importlib.util.spec_from_file_location(name, REPO / "experiments" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_sibling():
    name = "train_witness_realized_through_R_mlx"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REPO / "experiments" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ===========================================================================
# residual_compose module (pure numpy)
# ===========================================================================
def test_derive_composition_mask_selects_learn_classes():
    from tac.v2_compose.residual_compose import LEARN_CLASSES, derive_composition_mask

    assert LEARN_CLASSES == (1, 3)  # Lane, Movable (canonical comma10k order)
    lbl = np.array([[0, 1, 2, 3], [4, 0, 1, 3]], dtype=np.int64)
    m = derive_composition_mask(lbl, LEARN_CLASSES, dilate=0)
    expected = np.array([[0, 1, 0, 1], [0, 0, 1, 1]], dtype=bool)
    assert np.array_equal(m, expected), "mask must be exactly the Lane(1)+Movable(3) cells"
    assert m.dtype == bool


def test_derive_composition_mask_dilate_is_monotone_and_zero_is_exact():
    from tac.v2_compose.residual_compose import derive_composition_mask

    lbl = np.zeros((9, 9), np.int64)
    lbl[4, 4] = 1  # single lane pixel
    m0 = derive_composition_mask(lbl, (1, 3), dilate=0)
    m1 = derive_composition_mask(lbl, (1, 3), dilate=1)
    m2 = derive_composition_mask(lbl, (1, 3), dilate=2)
    assert int(m0.sum()) == 1
    assert int(m1.sum()) == 5  # 4-connectivity cross
    assert int(m2.sum()) == 13
    # dilation only grows the region
    assert np.all(m1[m0]) and np.all(m2[m1])


def test_compose_residual_rgb_where_semantics():
    from tac.v2_compose.residual_compose import compose_residual_rgb

    bulk = np.zeros((2, 3, 3), np.float32)
    inr = np.full((2, 3, 3), 255.0, np.float32)
    mask = np.array([[1, 0, 1], [0, 1, 0]], dtype=bool)
    comp = compose_residual_rgb(bulk, inr, mask)
    assert np.all(comp[mask] == 255.0), "INR shows through the mask"
    assert np.all(comp[~mask] == 0.0), "the deterministic bulk shows through elsewhere"


def test_residual_training_bundle_roundtrip(tmp_path):
    from tac.v2_compose.residual_compose import (
        build_residual_training_bundle,
        load_residual_training_bundle,
        save_residual_training_bundle,
    )

    rng = np.random.RandomState(0)
    bulk = (rng.rand(4, 6, 8, 3) * 255).astype(np.float32)
    labels = rng.randint(0, 5, (4, 6, 8)).astype(np.int64)
    b = build_residual_training_bundle(bulk, labels, dilate=1)
    p = tmp_path / "rb.npz"
    sz = save_residual_training_bundle(b, p)
    assert sz > 0 and p.exists() and (tmp_path / "rb.summary.json").exists()
    b2 = load_residual_training_bundle(p)
    assert np.array_equal(b.bulk_rgb_render_res, b2.bulk_rgb_render_res)
    assert np.array_equal(b.composition_mask, b2.composition_mask)
    assert b2.learn_classes == (1, 3) and b2.dilate == 1 and b2.n_pairs == 4


def test_save_bundle_refuses_tmp_path():
    from tac.v2_compose.residual_compose import build_residual_training_bundle, save_residual_training_bundle

    b = build_residual_training_bundle(np.zeros((1, 2, 2, 3), np.float32), np.zeros((1, 2, 2), np.int64))
    with pytest.raises(ValueError, match="transient /tmp"):
        save_residual_training_bundle(b, "/tmp/should_refuse.npz")


# ===========================================================================
# archive_grammar residual blob + byte accounting (numpy + brotli)
# ===========================================================================
def _tiny_inr_params(rng, in_feat, hidden, n_hidden, mod, n_classes=5):
    def lin(o, i):
        return (rng.standard_normal((o, i)) * 0.3).astype(np.float32)

    p = {
        "in_proj.weight": lin(hidden, in_feat), "in_proj.bias": (rng.standard_normal(hidden) * 0.1).astype(np.float32),
        "film.weight": lin(2 * hidden * n_hidden, mod), "film.bias": (rng.standard_normal(2 * hidden * n_hidden) * 0.1).astype(np.float32),
        "out_sdf.weight": lin(n_classes, hidden), "out_sdf.bias": (rng.standard_normal(n_classes) * 0.1).astype(np.float32),
        "out_tex.weight": lin(3, hidden), "out_tex.bias": (rng.standard_normal(3) * 0.1).astype(np.float32),
        "palette": (rng.rand(n_classes, 3) * 4 - 2).astype(np.float32),
        "code": (rng.standard_normal((6, mod)) * 0.5).astype(np.float32),
    }
    for li in range(n_hidden):
        p[f"hidden.{li}.weight"] = lin(hidden, hidden)
        p[f"hidden.{li}.bias"] = (rng.standard_normal(hidden) * 0.1).astype(np.float32)
    return p


def _tiny_inr_cfg(H, W, hidden, n_hidden, mod, bank, n_classes=5):
    return dict(
        n_hidden=n_hidden, hidden_dim=hidden, mod_dim=mod, n_classes=n_classes,
        activation="hosc", hosc_beta=4.0, hosc_omega=1.0, softmax_temp=0.1, wire_w0=20.0, wire_s0=10.0,
        chroma=True, render_h=H, render_w=W, bank_n_scales=bank["n_scales"], bank_n_orient0=bank["n_orient0"],
        bank_f0=bank["f0"], bank_base=bank["base"], bank_n_iso=bank["n_iso"], max_bank_freq=None,
        learn_classes=[1, 3], dilate=1)


def test_residual_blob_build_parse_roundtrip():
    from tac.boundary_math.lever_b_levelset_generator import CurveletBankConfig, curvelet_directional_B
    from tac.v2_compose.archive_grammar import build_residual_blob, parse_residual_blob

    bank = dict(n_scales=2, n_orient0=3, f0=2.0, base=2.0, n_iso=2)
    in_feat = 2 * curvelet_directional_B(CurveletBankConfig(**bank)).shape[1]
    rng = np.random.RandomState(3)
    params = _tiny_inr_params(rng, in_feat, hidden=6, n_hidden=2, mod=4)
    cfg = _tiny_inr_cfg(24, 32, 6, 2, 4, bank)
    blob = build_residual_blob(params, cfg)
    rb = parse_residual_blob(blob)
    # int8 dequant is lossy but tracks the original to within the per-tensor scale (sanity, not exact).
    for k in params:
        assert k in rb.params, f"{k} missing after parse"
        assert rb.params[k].shape == params[k].shape
        scale = float(np.abs(params[k]).max()) / 127.0 + 1e-6
        assert np.max(np.abs(rb.params[k] - params[k])) <= 2.0 * scale + 1e-5
    assert rb.manifest["learn_classes"] == [1, 3] and rb.manifest["dilate"] == 1


def _build_tiny_v2_packet(tmp_path, *, with_residual: bool, dilate: int = 1):
    """Assemble a tiny v2 archive (store + [residual] + pose) + return (packet_dir, oracle inputs)."""
    from tac.boundary_math.lever_b_levelset_generator import CurveletBankConfig, curvelet_directional_B
    from tac.v2_compose.archive_grammar import (
        assemble_v2_packet,
        build_residual_blob,
        build_store_blob,
        pack_v2_archive,
        parse_residual_blob,
        parse_store_blob,
    )

    rng = np.random.RandomState(11)
    H, W, n_pairs, n_classes = 24, 32, 3, 5
    reach_kstar = 2
    bank = dict(n_scales=2, n_orient0=3, f0=2.0, base=2.0, n_iso=2)
    keyframes = list(range(0, n_pairs, reach_kstar))
    kf_lstars = rng.randint(0, n_classes, (len(keyframes), H, W)).astype(np.int64)
    palette = (rng.rand(n_classes, 3) * 255).astype(np.float32)
    calib = (0.16, 0.05, 0.02)
    warp_codes = [0, 3, 2, 3, 1]
    store_blob = build_store_blob(keyframes, kf_lstars, palette, calib, warp_codes, reach_kstar, n_pairs, n_classes=n_classes)
    sb = parse_store_blob(store_blob)

    res_blob = b""
    rb = None
    if with_residual:
        in_feat = 2 * curvelet_directional_B(CurveletBankConfig(**bank)).shape[1]
        params = _tiny_inr_params(rng, in_feat, hidden=6, n_hidden=2, mod=4)
        cfg = _tiny_inr_cfg(H, W, 6, 2, 4, bank)
        cfg["dilate"] = dilate
        res_blob = build_residual_blob(params, cfg)
        rb = parse_residual_blob(res_blob)

    poses = (rng.standard_normal((n_pairs, 6)) * 0.05).astype(np.float32)
    comp = zlib.compress(poses.astype(np.float16).tobytes())
    pose_blob = (b"PNTG" + struct.pack("<H", 1) + struct.pack("<I", n_pairs)
                 + struct.pack("<I", 2 * n_pairs) + struct.pack("<I", len(comp)) + comp)
    poses_f16 = np.frombuffer(zlib.decompress(comp), dtype=np.float16).astype(np.float64).reshape(n_pairs, 6)

    blob = pack_v2_archive(store_blob, res_blob, pose_blob, b"{}")
    pkt = tmp_path / ("res" if with_residual else "floor")
    zip_path, _ = assemble_v2_packet(blob, pkt)
    (pkt / "0.bin").write_bytes(blob)
    return pkt, sb, rb, poses_f16, n_pairs, (store_blob, res_blob, pose_blob)


def test_byte_accounting_residual_present_and_floor(tmp_path):
    from tac.v2_compose.archive_grammar import byte_accounting

    pkt, _sb, _rb, _p, _n, (store_blob, res_blob, pose_blob) = _build_tiny_v2_packet(tmp_path, with_residual=True)
    acc = byte_accounting(pkt / "archive.zip", store_blob, res_blob, pose_blob, b"{}")
    assert acc["residual_inr_present"] is True
    assert acc["section_bytes"]["residual_inr_blob"] > 0
    assert acc["archive_zip_bytes"] > 0 and 0.0 < acc["rate"] < 1.0

    pkt2, _sb2, _rb2, _p2, _n2, (sb2, rb2, pb2) = _build_tiny_v2_packet(tmp_path, with_residual=False)
    acc2 = byte_accounting(pkt2 / "archive.zip", sb2, rb2, pb2, b"{}")
    assert acc2["residual_inr_present"] is False
    assert acc2["section_bytes"]["residual_inr_blob"] == 0


# ===========================================================================
# INFLATE PARITY (the NO-FAKE faithfulness anchor) -- subprocess inflate vs numpy oracle.
# ===========================================================================
@pytest.mark.skipif(not _HAS_TORCH, reason="torch (bicubic R) unavailable")
def test_inflate_residual_compose_bit_exact_parity(tmp_path):
    from tac.v2_compose.archive_grammar import residual_inflate_reference

    pkt, sb, rb, poses_f16, n_pairs, _blobs = _build_tiny_v2_packet(tmp_path, with_residual=True, dilate=1)
    dst = pkt / "0.raw"
    r = subprocess.run([sys.executable, str(pkt / "inflate.py"), str(pkt / "0.bin"), str(dst)],
                       capture_output=True, text=True)
    assert r.returncode == 0, f"inflate failed: {r.stderr}"
    raw = np.fromfile(dst, dtype=np.uint8).reshape(2 * n_pairs, 874, 1164, 3)
    oracle = residual_inflate_reference(sb, rb, poses_f16, n_pairs)
    assert raw.shape == oracle.shape
    assert np.array_equal(raw, oracle), (
        "inflate residual-compose MUST be bit-identical to the numpy oracle "
        f"(max abs diff {int(np.abs(raw.astype(int) - oracle.astype(int)).max())})")


@pytest.mark.skipif(not _HAS_TORCH, reason="torch (bicubic R) unavailable")
def test_inflate_deterministic_floor_empty_residual(tmp_path):
    pkt, _sb, _rb, _p, n_pairs, _blobs = _build_tiny_v2_packet(tmp_path, with_residual=False)
    dst = pkt / "0.raw"
    r = subprocess.run([sys.executable, str(pkt / "inflate.py"), str(pkt / "0.bin"), str(dst)],
                       capture_output=True, text=True)
    assert r.returncode == 0, f"floor inflate failed: {r.stderr}"
    raw = np.fromfile(dst, dtype=np.uint8).reshape(2 * n_pairs, 874, 1164, 3)
    # empty residual => bulk-only => frame0 == frame1 per pair (the deterministic floor).
    for p in range(n_pairs):
        assert np.array_equal(raw[2 * p], raw[2 * p + 1]), "floor: f0 must equal f1 (bulk only)"


# ===========================================================================
# render seam (sibling): compose_fn=None is byte-identical; a compose_fn composes.
# ===========================================================================
@pytest.mark.skipif(not _HAS_MLX, reason="mlx not available")
def test_render_through_R_compose_none_is_byte_identical():
    import mlx.core as mx

    from tac.local_acceleration.mlx_scorer_adapters import temporary_mlx_device

    sib = _load_sibling()
    lst = _load_levelset_trainer()
    rng = np.random.RandomState(5)
    in_feat = 12
    # CPU device throughout: NEVER contend with the live GPU baseline (hard rail).
    with temporary_mlx_device("cpu"):
        model = lst.build_levelset_rgb_witness(
            num_pairs=2, in_feat=in_feat, hidden_dim=8, n_hidden=2, mod_dim=4, n_classes=5,
            activation="hosc", softmax_temp=0.1, wire_w0=20.0, wire_s0=10.0, hosc_beta=4.0, hosc_omega=1.0,
            chroma=True)
        mx.eval(model.parameters())
        h, w = 48, 64
        feats = mx.array((rng.standard_normal((h * w, in_feat)) * 0.5).astype(np.float32))
        bare = np.asarray(sib.render_through_R_mlx(model, feats, 0, h, w))
        none = np.asarray(sib.render_through_R_mlx(model, feats, 0, h, w, compose_fn=None))
    assert np.array_equal(bare, none), "compose_fn=None MUST be byte-identical to the bare render"


@pytest.mark.skipif(not _HAS_MLX, reason="mlx not available")
def test_compose_fn_composes_bulk_and_inr():
    import mlx.core as mx

    from tac.local_acceleration.mlx_scorer_adapters import temporary_mlx_device

    sib = _load_sibling()
    lst = _load_levelset_trainer()
    rng = np.random.RandomState(6)
    in_feat = 12
    h, w = 48, 64
    with temporary_mlx_device("cpu"):  # NEVER contend with the live GPU baseline (hard rail).
        model = lst.build_levelset_rgb_witness(
            num_pairs=2, in_feat=in_feat, hidden_dim=8, n_hidden=2, mod_dim=4, n_classes=5,
            activation="hosc", softmax_temp=0.1, wire_w0=20.0, wire_s0=10.0, hosc_beta=4.0, hosc_omega=1.0,
            chroma=True)
        mx.eval(model.parameters())
        feats = mx.array((rng.standard_normal((h * w, in_feat)) * 0.5).astype(np.float32))
        bulk = mx.array((rng.rand(1, h, w, 3) * 255).astype(np.float32))

        # mask all-1 -> INR shows through everywhere -> identical to the bare render.
        def compose_all_inr(rgb, code_idx):
            return rgb

        # mask all-0 -> bulk shows through everywhere -> identical to R(bulk).
        def compose_all_bulk(rgb, code_idx):
            return bulk

        bare = np.asarray(sib.render_through_R_mlx(model, feats, 1, h, w))
        via_inr = np.asarray(sib.render_through_R_mlx(model, feats, 1, h, w, compose_fn=compose_all_inr))
        via_bulk = np.asarray(sib.render_through_R_mlx(model, feats, 1, h, w, compose_fn=compose_all_bulk))

        # a trivial "witness" that just emits the fixed bulk -> R(bulk) reference.
        class _BulkWitness:
            def __call__(self, _feats, _idx):
                return mx.reshape(bulk, (h * w, 3))

        r_bulk = np.asarray(sib.render_through_R_mlx(_BulkWitness(), feats, 0, h, w))

    assert np.array_equal(via_inr, bare), "mask=1 compose must equal the INR render"
    assert np.array_equal(via_bulk, r_bulk), "mask=0 compose must equal R(bulk)"
    assert not np.array_equal(via_bulk, bare), "bulk and INR renders must actually differ"


@pytest.mark.skipif(not (_HAS_MLX and _scorer_weights_present()), reason="mlx + upstream scorers required")
def test_make_loss_fn_render_fn_default_is_identical():
    """make_loss_fn(render_fn=None) MUST produce the SAME loss as render_fn=render_through_R_mlx
    (the baseline-path-unchanged proof at the loss level: the default render IS the bare render)."""
    import mlx.core as mx

    from tac.local_acceleration.mlx_scorer_adapters import (
        load_mlx_distortion_scorer_adapter_from_upstream,
        temporary_mlx_device,
    )

    sib = _load_sibling()
    lst = _load_levelset_trainer()
    rng = np.random.RandomState(9)
    in_feat, h, w = 12, 96, 128
    with temporary_mlx_device("cpu"):
        adapter = load_mlx_distortion_scorer_adapter_from_upstream(REPO / "upstream", device="cpu")
        model = lst.build_levelset_rgb_witness(
            num_pairs=1, in_feat=in_feat, hidden_dim=8, n_hidden=2, mod_dim=4, n_classes=5,
            activation="hosc", softmax_temp=0.1, wire_w0=20.0, wire_s0=10.0, hosc_beta=4.0, hosc_omega=1.0,
            chroma=True)
        mx.eval(model.parameters())
        feats = mx.array((rng.standard_normal((h * w, in_feat)) * 0.5).astype(np.float32))
        oh = np.zeros((1, sib.SEG_H, sib.SEG_W, 5), np.float32)
        oh[0, ..., 0] = 1.0
        lstar_oh = mx.array(oh)
        margin = mx.zeros((sib.SEG_H, sib.SEG_W))
        pose_tgt = mx.zeros((6,))
        args = (model, feats, 0, 1, lstar_oh, margin, pose_tgt,
                mx.array(100.0), mx.array(0.0), mx.array(4.0), mx.array(0.5))
        loss_default = float(sib.make_loss_fn(adapter, h, w)(*args))
        loss_explicit = float(sib.make_loss_fn(adapter, h, w, render_fn=sib.render_through_R_mlx)(*args))
    assert loss_default == loss_explicit, "render_fn=None must be the bare render (baseline unchanged)"


# ===========================================================================
# coordinator coherence: a surgical lever reweights the COMPOSED-render seg loss.
# ===========================================================================
@pytest.mark.skipif(not (_HAS_MLX and _scorer_weights_present()), reason="mlx + upstream scorers required")
def test_surgical_lever_reweights_composed_loss():
    import mlx.core as mx

    from tac.boundary_math.lever_b_levelset_generator import lane_thin_weight_map
    from tac.local_acceleration.mlx_scorer_adapters import (
        load_mlx_distortion_scorer_adapter_from_upstream,
        temporary_mlx_device,
    )

    sib = _load_sibling()
    lst = _load_levelset_trainer()
    SEG_H, SEG_W = sib.SEG_H, sib.SEG_W
    rng = np.random.RandomState(8)
    in_feat = 12
    h, w = 96, 128
    with temporary_mlx_device("cpu"):
        adapter = load_mlx_distortion_scorer_adapter_from_upstream(REPO / "upstream", device="cpu")
        model = lst.build_levelset_rgb_witness(
            num_pairs=1, in_feat=in_feat, hidden_dim=8, n_hidden=2, mod_dim=4, n_classes=5,
            activation="hosc", softmax_temp=0.1, wire_w0=20.0, wire_s0=10.0, hosc_beta=4.0, hosc_omega=1.0,
            chroma=True)
        mx.eval(model.parameters())
        feats = mx.array((rng.standard_normal((h * w, in_feat)) * 0.5).astype(np.float32))
        bulk = mx.array((rng.rand(1, h, w, 3) * 255).astype(np.float32))
        # a non-trivial composition mask (half INR, half bulk) at render res.
        mask_np = np.zeros((h, w, 1), np.float32)
        mask_np[:, : w // 2] = 1.0
        mask_mx = mx.array(mask_np)[None]  # (1,h,w,1)

        def compose_fn(rgb, code_idx):
            return bulk * (1.0 - mask_mx) + rgb * mask_mx

        def _render_R(witness, cf, idx, rh, rw):
            return sib.render_through_R_mlx(witness, cf, idx, rh, rw, compose_fn=compose_fn)

        # the COMPOSED realized seg logits (== what the lever consumes in residual mode).
        f1_composed = _render_R(model, feats, 1, h, w)            # (1,SEG_H,SEG_W,3)
        f1_bare = sib.render_through_R_mlx(model, feats, 1, h, w)
        assert not np.allclose(np.asarray(f1_composed), np.asarray(f1_bare)), (
            "compose_fn MUST change the realized render (the lever runs on the COMPOSED witness)")
        slog = adapter.segnet(f1_composed)                        # (1,SEG_H,SEG_W,5)

        # GT one-hot with a real lane (class 1) band so the lane-thin map is non-trivial.
        lstar = np.zeros((SEG_H, SEG_W), np.int64)
        lstar[:, SEG_W // 3: SEG_W // 3 + 4] = 1                  # a thin vertical lane stripe
        oh = np.zeros((1, SEG_H, SEG_W, 5), np.float32)
        for c in range(5):
            oh[0, ..., c] = (lstar == c).astype(np.float32)
        lstar_oh = mx.array(oh)
        # the lever's realized decision margin (gt_logit - top competitor), exactly as total_loss_fn.
        sig_gt = mx.sum(slog * lstar_oh, axis=-1)
        sig_run = mx.max(slog + lstar_oh * (-1e9), axis=-1)
        signed = sig_gt - sig_run                                  # (1,SEG_H,SEG_W)
        hinge = mx.maximum(0.5 - signed, 0.0)                     # relu(target - margin)
        thin = mx.array(lane_thin_weight_map(lstar, lane_class=1, radius=4)[None])  # (1,SEG_H,SEG_W)

        unweighted = float(mx.mean(hinge))
        weighted = float(mx.mean(hinge * (1.0 + 2.0 * thin)))     # lane-thin up-weight (LEVER-B)
        assert float(thin.sum()) > 0.0, "the lane-thin map must be non-trivial (real lane present)"
        assert abs(weighted - unweighted) > 1e-6, (
            "the lane-thin lever MUST measurably reweight the COMPOSED-render seg loss "
            "(not a silent no-op) -- this is what lets the theta* sweep run on the hybrid residual")


# ===========================================================================
# trainer residual-mode fail-closed config guards (fast: import + main(argv), pre-run_train).
# ===========================================================================
@pytest.mark.skipif(not _HAS_MLX, reason="mlx not available (trainer import)")
def test_residual_mode_requires_npz():
    lst = _load_levelset_trainer()
    with pytest.raises(ValueError, match="requires --residual-target-npz"):
        lst.main(["--out-dir", str(REPO / "experiments" / "results" / "_nope"), "--residual-mode"])


@pytest.mark.skipif(not _HAS_MLX, reason="mlx not available (trainer import)")
def test_residual_mode_incompatible_with_structured_init():
    lst = _load_levelset_trainer()
    with pytest.raises(ValueError, match="incompatible with --structured-init"):
        lst.main(["--out-dir", str(REPO / "experiments" / "results" / "_nope"),
                  "--residual-mode", "--residual-target-npz", "b.npz", "--structured-init"])


@pytest.mark.skipif(not _HAS_MLX, reason="mlx not available (trainer import)")
def test_residual_npz_without_mode_fails():
    lst = _load_levelset_trainer()
    with pytest.raises(ValueError, match="--residual-mode is OFF"):
        lst.main(["--out-dir", str(REPO / "experiments" / "results" / "_nope"),
                  "--residual-target-npz", "b.npz"])


@pytest.mark.skipif(not _HAS_MLX, reason="mlx not available (trainer import)")
def test_residual_mode_incompatible_with_freeze_decoder():
    lst = _load_levelset_trainer()
    with pytest.raises(ValueError, match="incompatible with --freeze-decoder-fit-codes"):
        lst.main(["--out-dir", str(REPO / "experiments" / "results" / "_nope"),
                  "--residual-mode", "--residual-target-npz", "b.npz",
                  "--freeze-decoder-fit-codes", "d.npz"])
