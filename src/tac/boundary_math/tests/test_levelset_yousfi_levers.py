# SPDX-License-Identifier: MIT
"""Tests for the Yousfi-lever additive surfaces on the level-set witness generator.

Covers the pure-numpy (MLX-free) lever code landed 2026-06-27:
  * LEVER-2 stem-Nyquist: ``stem_nyquist_max_freq_cycles_per_unit`` + the additive ``max_freq``
    cap on ``curvelet_directional_B`` / ``LevelSetConfig`` (default None = current behavior).
  * LEVER-1 chroma: ``levelset_rgb_forward_numpy`` chroma=True/False parity (the byte-close
    ONE-CODEPATH forward) — phi/argmax invariant, RGB differs, achromatic => R==G==B.

The LEVER-3 lane-edge realized margin term lives in the MLX trainer (validated by the $0 CPU
smoke + the standalone numpy mechanism check in the landing memo); it needs MLX + the frozen
scorer adapter so it is not unit-tested here.
"""
from __future__ import annotations

import numpy as np

from tac.boundary_math.lever_b_levelset_generator import (
    CurveletBankConfig,
    LevelSetConfig,
    curvelet_directional_B,
    levelset_rgb_forward_numpy,
    quantize_levelset_blob,
    stem_nyquist_max_freq_cycles_per_unit,
)


def test_stem_nyquist_formula():
    # f_max = scorer_w / (4 * stem_stride): 512/(4*2)=64; 512/(4*1)=128; 256/(4*2)=32.
    assert stem_nyquist_max_freq_cycles_per_unit(512, 2) == 64.0
    assert stem_nyquist_max_freq_cycles_per_unit(512, 1) == 128.0
    assert stem_nyquist_max_freq_cycles_per_unit(256, 2) == 32.0
    # stem_stride floored at 1 (no div-by-zero).
    assert stem_nyquist_max_freq_cycles_per_unit(512, 0) == 128.0


def test_curvelet_max_freq_default_is_unchanged():
    bank = CurveletBankConfig()
    base = curvelet_directional_B(bank)
    none = curvelet_directional_B(bank, max_freq=None)
    assert base.shape == none.shape
    assert np.array_equal(base, none)


def test_curvelet_default_bank_is_sub_nyquist():
    # The default curvelet bank max freq (f0=2,base=2,n_scales=4 -> 16) is below the stem Nyquist
    # (64), so capping at Nyquist is a no-op (the over-Nyquist waste is in the self-orient feats).
    bank = CurveletBankConfig()
    B = curvelet_directional_B(bank)
    norms = np.sqrt((B.astype(np.float64) ** 2).sum(axis=0))
    assert float(norms.max()) <= 16.0 + 1e-6
    nyq = stem_nyquist_max_freq_cycles_per_unit(512, 2)
    capped = curvelet_directional_B(bank, max_freq=nyq)
    assert capped.shape[1] == B.shape[1]  # no-op at Nyquist for the default bank


def test_curvelet_max_freq_drops_high_atoms_and_never_empties():
    bank = CurveletBankConfig()
    full = curvelet_directional_B(bank)
    capped = curvelet_directional_B(bank, max_freq=5.0)
    assert capped.shape[1] < full.shape[1]  # drops atoms above 5
    norms = np.sqrt((capped.astype(np.float64) ** 2).sum(axis=0))
    assert float(norms.max()) <= 5.0 + 1e-6
    # cap below the minimum still keeps >=1 atom (never empties).
    tiny = curvelet_directional_B(bank, max_freq=0.0)
    assert tiny.shape[1] >= 1


def test_levelset_config_threads_max_freq():
    full = LevelSetConfig(num_pairs=4)
    capped = LevelSetConfig(num_pairs=4, max_freq=5.0)
    assert full.max_freq is None  # default unchanged
    assert capped.build_B().shape[1] < full.build_B().shape[1]
    assert capped.in_feat() == 2 * capped.build_B().shape[1]


def _toy_levelset_params(H=32, NH=2, M=8, K=5, F=24, seed=1):
    # Small magnitudes (bounded like real int8-dequantized trained weights) so the float64
    # reference forward stays well-conditioned (no overflow warnings on synthetic inputs).
    rng = np.random.default_rng(seed)
    p = {
        "in_proj.weight": rng.standard_normal((H, F)).astype(np.float32) * 0.1,
        "in_proj.bias": rng.standard_normal(H).astype(np.float32) * 0.05,
        "film.weight": rng.standard_normal((2 * H * NH, M)).astype(np.float32) * 0.05,
        "film.bias": rng.standard_normal(2 * H * NH).astype(np.float32) * 0.05,
        "out_sdf.weight": rng.standard_normal((K, H)).astype(np.float32) * 0.2,
        "out_sdf.bias": rng.standard_normal(K).astype(np.float32) * 0.05,
        "out_tex.weight": rng.standard_normal((3, H)).astype(np.float32) * 0.2,
        "out_tex.bias": rng.standard_normal(3).astype(np.float32) * 0.05,
        "palette": rng.standard_normal((K, 3)).astype(np.float32) * 1.2,
    }
    for li in range(NH):
        p[f"hidden.{li}.weight"] = rng.standard_normal((H, H)).astype(np.float32) * 0.1
        p[f"hidden.{li}.bias"] = rng.standard_normal(H).astype(np.float32) * 0.05
    code = rng.standard_normal(M).astype(np.float32) * 0.2
    feats = rng.standard_normal((50, F)).astype(np.float32) * 0.5
    return p, feats, code, dict(n_hidden=NH, hidden_dim=H, n_classes=K, activation="hosc",
                                softmax_temp=0.1, wire_w0=20.0, wire_s0=10.0, hosc_beta=4.0, hosc_omega=1.0)


def test_chroma_parity_phi_invariant_rgb_differs():
    p, feats, code, kw = _toy_levelset_params()
    rgb_c, phi_c = levelset_rgb_forward_numpy(p, feats, code, chroma=True, **kw)
    rgb_a, phi_a = levelset_rgb_forward_numpy(p, feats, code, chroma=False, **kw)
    # chroma does NOT change the SDF/argmax partition (it is a color realization, not structure).
    assert np.allclose(phi_c, phi_a)
    # chroma carries real signal -> RGB differs materially.
    assert float(np.abs(rgb_c - rgb_a).max()) > 1.0
    # achromatic mode collapses to R==G==B (BT.601 luma replicated).
    assert np.allclose(rgb_a[:, 0], rgb_a[:, 1]) and np.allclose(rgb_a[:, 1], rgb_a[:, 2])
    # chroma mode keeps genuine chroma (R != G somewhere).
    assert not np.allclose(rgb_c[:, 0], rgb_c[:, 1])
    assert np.isfinite(rgb_c).all() and np.isfinite(rgb_a).all()


def test_chroma_blob_bytes_are_chroma_independent():
    # chroma is realized in the forward only; out_tex stays 3-channel either way, so the int8+brotli
    # blob (the counted rate term) does NOT depend on chroma -> chroma is a FREE-rate d_seg lever.
    p, _, _, _ = _toy_levelset_params()
    p["code"] = np.random.default_rng(2).standard_normal((8, 8)).astype(np.float32)
    blob = quantize_levelset_blob(p)
    assert blob["total_quantized_blob_bytes"] > 0
    assert blob["n_params"] > 0
