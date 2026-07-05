# SPDX-License-Identifier: MIT
"""Tests for the #268 S_R reachability weight stack:

* ``tac.boundary_math.sr_rchain_gain`` — the exact linear R-chain column-norm derivation
  (numpy reference; torch-parity; Kronecker outer-product identity; determinism/sha).
* ``tools/precompute_sR_reachability.py`` — normalization + cache-write contracts.
* trainer/probe argparse + fail-closed contracts (source-scan per the NEVER-invent-flags
  discipline; the trainer module is too heavy to import in a unit test).

Axis: pure unit tests — no score claims. Pointer 0.19110 UNMOVED.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]

from tac.boundary_math.sr_rchain_gain import (  # noqa: E402
    CAMERA_HW,
    SEG_HW,
    bicubic_resample_matrix_1d,
    bilinear_resample_matrix_1d,
    rchain_1d_operator,
    rchain_column_l1_map,
    rchain_column_l1_profiles,
    rchain_map_sha256,
    rchain_signed_colsum_map,
)


def _load_tool():
    spec = importlib.util.spec_from_file_location(
        "precompute_sR_reachability", REPO / "tools" / "precompute_sR_reachability.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("precompute_sR_reachability", mod)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- 1-D matrix invariants
def test_bilinear_rows_sum_to_one():
    M = bilinear_resample_matrix_1d(97, 41)
    np.testing.assert_allclose(M.sum(axis=1), 1.0, atol=1e-12)


def test_bicubic_rows_sum_to_one():
    # cubic-convolution weights are a partition of unity at every fractional offset
    M = bicubic_resample_matrix_1d(41, 97)
    np.testing.assert_allclose(M.sum(axis=1), 1.0, atol=1e-12)


def test_bicubic_matrix_matches_torch():
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F

    n_in, n_out = 96, 218
    rng = np.random.default_rng(1)
    x = rng.standard_normal(n_in)
    M = bicubic_resample_matrix_1d(n_in, n_out)
    t = torch.from_numpy(np.tile(x[:, None], (1, 4))[None, None]).double()
    y = F.interpolate(t, size=(n_out, 4), mode="bicubic", align_corners=False)[0, 0, :, 2].numpy()
    np.testing.assert_allclose(M @ x, y, atol=1e-12)


def test_bilinear_matrix_matches_torch():
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F

    n_in, n_out = 218, 96
    rng = np.random.default_rng(2)
    x = rng.standard_normal(n_in)
    M = bilinear_resample_matrix_1d(n_in, n_out)
    t = torch.from_numpy(np.tile(x[:, None], (1, 4))[None, None]).double()
    y = F.interpolate(t, size=(n_out, 4), mode="bilinear", align_corners=False)[0, 0, :, 2].numpy()
    np.testing.assert_allclose(M @ x, y, atol=1e-12)


# ------------------------------------------------- Kronecker outer-product exact identity
def test_l1_column_norm_outer_product_identity_vs_dense_kron():
    # Tiny sizes: build the FULL 2-D operator via np.kron and verify the closed form
    # ||column_(r,c)|D||_1 == a_v[r]*a_h[c] EXACTLY (the derivation the module documents).
    seg_hw, cam_hw = (8, 10), (18, 23)
    D_v = rchain_1d_operator(seg_hw[0], cam_hw[0])
    D_h = rchain_1d_operator(seg_hw[1], cam_hw[1])
    D2 = np.kron(D_v, D_h)  # (8*10, 8*10) full 2-D operator
    dense_l1 = np.abs(D2).sum(axis=0).reshape(seg_hw)
    closed = rchain_column_l1_map(seg_hw, cam_hw)
    np.testing.assert_allclose(dense_l1, closed, atol=1e-12)


def test_signed_colsum_map_matches_torch_vjp_of_ones():
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F

    seg_hw, cam_hw = (48, 64), (109, 146)
    x = torch.zeros(1, 1, *seg_hw, dtype=torch.float64, requires_grad=True)
    up = F.interpolate(x, size=cam_hw, mode="bicubic", align_corners=False)
    dn = F.interpolate(up, size=seg_hw, mode="bilinear", align_corners=False)
    dn.sum().backward()
    np.testing.assert_allclose(
        x.grad[0, 0].numpy(), rchain_signed_colsum_map(seg_hw, cam_hw), atol=1e-12)


def test_abs_column_norm_matches_torch_delta_forward():
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F

    seg_hw, cam_hw = (48, 64), (109, 146)
    closed = rchain_column_l1_map(seg_hw, cam_hw)
    for (r, c) in [(0, 0), (5, 7), (24, 32), (47, 63)]:
        d = torch.zeros(1, 1, *seg_hw, dtype=torch.float64)
        d[0, 0, r, c] = 1.0
        u = F.interpolate(d, size=cam_hw, mode="bicubic", align_corners=False)
        dd = F.interpolate(u, size=seg_hw, mode="bilinear", align_corners=False)
        assert abs(float(dd.abs().sum()) - float(closed[r, c])) < 1e-12


# --------------------------------------------------------- static map: shape/determinism
def test_static_map_is_pair_and_content_independent_by_construction():
    # The API takes NO frame/pair/theta input — pair-independence is structural. Verify
    # shape, positivity, finiteness at contest sizes.
    m = rchain_column_l1_map()
    assert m.shape == SEG_HW
    assert np.all(np.isfinite(m)) and np.all(m > 0)


def test_static_map_deterministic_and_sha_stable():
    s1 = rchain_map_sha256()
    s2 = rchain_map_sha256()
    assert s1 == s2 and len(s1) == 64
    m1 = rchain_column_l1_map()
    m2 = rchain_column_l1_map()
    assert m1.tobytes() == m2.tobytes()


def test_static_map_measured_band_pinned():
    # Measured 2026-07-05 (sr_geo_analysis): mean 1.07200, ripple +-6.6%, range [0.933, 1.272].
    # Loose 1e-6 pins document the derivation output; exact bytes are BLAS-order dependent
    # so the pin is on values, not the sha.
    m = rchain_column_l1_map()
    assert abs(float(m.mean()) - 1.0719983725538544) < 1e-6
    assert abs(float(m.min()) - 0.9330917312184508) < 1e-6
    assert abs(float(m.max()) - 1.271934914254838) < 1e-6
    a_v, a_h = rchain_column_l1_profiles()
    assert a_v.shape == (SEG_HW[0],) and a_h.shape == (SEG_HW[1],)
    np.testing.assert_allclose(np.outer(a_v, a_h), m, atol=1e-12)


def test_camera_hw_constants_match_contest():
    assert CAMERA_HW == (874, 1164) and SEG_HW == (384, 512)


# ------------------------------------------------------------------ tool: normalization
def test_normalize_sR_range_and_p99_clip():
    tool = _load_tool()
    rng = np.random.default_rng(3)
    raw = rng.exponential(1.0, size=(64, 64)).astype(np.float32)
    out = tool._normalize_sR(raw, pct=99.0)
    assert out.dtype == np.float32
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0
    # everything above p99 saturates at exactly 1.0 (robust, outlier-safe)
    thr = np.percentile(raw, 99.0)
    assert np.all(out[raw > thr + 1e-6] == 1.0)


def test_normalize_sR_constant_map_edge_case():
    tool = _load_tool()
    out = tool._normalize_sR(np.full((8, 8), 3.5, np.float32))
    assert np.all(np.isfinite(out)) and np.all(out <= 1.0) and np.all(out > 0.99)


def test_normalize_sR_zero_map_edge_case():
    tool = _load_tool()
    out = tool._normalize_sR(np.zeros((8, 8), np.float32))
    assert np.all(out == 0.0)


# ------------------------------------------------------------------ tool: cache writes
def _mk_cache(path: Path) -> dict[str, np.ndarray]:
    arrays = {
        "n_pairs": np.int64(2),
        "gt_f1": np.arange(2 * 4 * 5 * 3, dtype=np.uint8).reshape(2, 4, 5, 3),
        "margins": np.ones((2, 4, 5), np.float32),
    }
    with open(path, "wb") as fh:
        np.savez(fh, **arrays)
    return arrays


def test_write_sidecar_never_touches_main_cache(tmp_path):
    tool = _load_tool()
    cache = tmp_path / "gt_tiny.npz"
    _mk_cache(cache)
    before = cache.read_bytes()
    sR = np.random.default_rng(4).random((2, 4, 5)).astype(np.float32)
    sidecar = tool.write_sidecar(cache, sR)
    assert cache.read_bytes() == before  # main cache bit-untouched
    z = np.load(sidecar)
    assert set(z.files) == {"n_pairs", "sR"}
    np.testing.assert_array_equal(z["sR"], sR)


def test_write_inplace_preserves_existing_members_and_adds_sR(tmp_path):
    tool = _load_tool()
    cache = tmp_path / "gt_tiny.npz"
    orig = _mk_cache(cache)
    sR = np.random.default_rng(5).random((2, 4, 5)).astype(np.float32)
    tool.write_inplace(cache, sR)
    z = np.load(cache)
    assert set(z.files) == {"n_pairs", "gt_f1", "margins", "sR"}
    for k, v in orig.items():
        np.testing.assert_array_equal(z[k], v)  # existing members preserved exactly
    np.testing.assert_array_equal(z["sR"], sR)


def test_tool_refuses_tmp_class_durable_paths():
    tool = _load_tool()
    with pytest.raises(ValueError, match="tmp-class"):
        tool._refuse_tmp(Path("/tmp/gt_n600.npz"))
    with pytest.raises(ValueError, match="tmp-class"):
        tool._refuse_tmp(Path("/private/tmp/x/gt.npz"))


# ------------------------------------------------ trainer/probe contracts (source scan)
def test_trainer_flag_exists_default_off_and_fails_closed():
    src = (REPO / "experiments" / "train_levelset_witness_realized_through_R_mlx.py").read_text()
    i = src.index('"--margin-saliency-reachability"')
    assert 'action="store_true"' in src[i:i + 400]  # boolean, default OFF
    # fail-closed contracts (never silent): micro-batch and missing-sR both raise
    assert "not supported with --micro-batch-pairs" in src
    assert "has no 'sR' key" in src
    # OFF path never references the provider (byte-identity precondition)
    assert "msal_reach and _sR_provider is not None" in src


def test_trainer_sidecar_fallback_contract():
    # #268 sidecar fallback: precedence main-cache 'sR' > '<stem>_sR.npz' sidecar > fail closed,
    # INSIDE the msal_reach gate (OFF path untouched — byte-identity proven bitwise on n6:
    # A2==A and F(sidecar)==E(main-cache) modulo the __cfg_git_sha provenance leaf, 2026-07-05).
    src = (REPO / "experiments" / "train_levelset_witness_realized_through_R_mlx.py").read_text()
    assert '_sR.npz' in src and "SIDECAR FALLBACK (#268)" in src
    assert '"sR_source": _sR_src' in src  # telemetry reports which source fed the provider
    i = src.index("SIDECAR FALLBACK (#268)")
    gate = src.rindex("if msal_reach and msal_w > 0.0:", 0, i)
    assert i - gate < 2500  # fallback lives inside the flag-gated populate block


def test_probe_argparse_contract_matches_tool_surface():
    src = (REPO / "experiments" / "probe_sr_reachability_calibration.py").read_text()
    for flag in ("--ckpt", "--gt-cache", "--out-json", "--pairs", "--msal-tau",
                 "--msal-target", "--uniward-beta"):
        assert f'"{flag}"' in src, flag
    assert "has no 'sR' key" in src  # probe fails closed without the cached map too
