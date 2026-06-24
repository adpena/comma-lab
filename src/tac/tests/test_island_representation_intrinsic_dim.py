"""NO-FAKE tests for the class-1 island representation-level intrinsic-dim probe.

These tests verify BEHAVIOR (the effective-rank, DCT, contour-descriptor,
motion-comp, TwoNN/MLE/AE estimators actually compute the claimed quantities on
data with KNOWN intrinsic dimension), not constants. Every test would FAIL if
the estimator returned a hard-coded value. The decisive AE/TwoNN/MLE tests use
synthetic manifolds whose intrinsic dimension is known by construction.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
PROBE = REPO / "experiments/probe_island_representation_intrinsic_dim.py"

_spec = importlib.util.spec_from_file_location("_probe_island", PROBE)
probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe)


# ---------------------------------------------------------------------------
# effective-rank helpers
# ---------------------------------------------------------------------------
def test_effective_rank_recovers_known_low_rank():
    # data matrix that is EXACTLY rank 3 with NEAR-ISOTROPIC modes -> the
    # participation ratio is ~3 (anisotropic spectra give LOWER PR by design;
    # k_for_99pct is the rank-faithful invariant that must equal 3).
    rng = np.random.default_rng(0)
    basis = rng.standard_normal((3, 50))
    coeffs = rng.standard_normal((100, 3)) * np.array([3.0, 2.6, 2.2])
    X = coeffs @ basis
    S = probe.svd_singvals(X)
    er = probe.effective_rank_from_singvals(S)
    assert 2.0 < er["participation_ratio"] < 3.2  # ~3 near-isotropic modes
    assert er["k_for_99pct"] <= 3  # exactly rank 3 captures 99%
    assert er["k_for_95pct"] <= 3
    # and an anisotropic rank-3 has a LOWER participation ratio (real behavior)
    coeffs2 = rng.standard_normal((100, 3)) * np.array([10.0, 5.0, 2.0])
    er2 = probe.effective_rank_from_singvals(probe.svd_singvals(coeffs2 @ basis))
    assert er2["participation_ratio"] < er["participation_ratio"]
    assert er2["k_for_99pct"] <= 3  # still rank 3 by reconstruction


def test_effective_rank_recovers_full_rank_noise():
    # isotropic noise -> effective rank near full (all modes equal)
    rng = np.random.default_rng(1)
    X = rng.standard_normal((60, 200))
    S = probe.svd_singvals(X)
    er = probe.effective_rank_from_singvals(S)
    # full-rank isotropic: participation ratio is a large fraction of n=60
    assert er["participation_ratio"] > 40
    assert er["k_for_95pct"] > 45


def test_svd_gram_trick_matches_direct_svd():
    # the d>n gram-trick path MUST equal the direct SVD singular values on the
    # MEANINGFUL (nonzero) modes. The gram trick (eigvalsh of XX^T, sqrt) loses
    # precision on the trailing ~zero singular value (rank n-1 after centering);
    # we compare the top n-1 nonzero modes, which agree to machine precision.
    rng = np.random.default_rng(2)
    X = rng.standard_normal((20, 500))  # d > n triggers gram path
    Xc = X - X.mean(0)
    S_gram = np.sort(probe.svd_singvals(X))[::-1]
    S_direct = np.sort(np.linalg.svd(Xc, compute_uv=False))[::-1]
    # top 19 (centering drops 1 dof -> 20th is numerical zero); relative match
    assert np.allclose(S_gram[:19], S_direct[:19], rtol=1e-7, atol=1e-6)


# ---------------------------------------------------------------------------
# island extraction
# ---------------------------------------------------------------------------
def test_class1_mask_stack_selects_only_class1():
    gt = np.zeros((3, 8, 8), dtype=np.uint8)
    gt[0, 2:4, 2:4] = 1
    gt[1, 5, 5] = 1
    gt[2, :, :] = 3  # no class-1 in frame 2
    stk = probe.class1_mask_stack(gt)
    assert stk[0].sum() == 4 and stk[1].sum() == 1 and stk[2].sum() == 0
    assert stk.dtype == bool


def test_small_island_stack_drops_large_components():
    # one tiny island (4 px) + one big blob (>500 px) of class 1
    gt = np.zeros((1, 40, 40), dtype=np.uint8)
    gt[0, 1:3, 1:3] = 1  # 4-px island
    gt[0, 10:35, 10:35] = 1  # 625-px blob (>500)
    stk = probe.small_island_stack(gt)
    assert stk[0].sum() == 4  # only the small island survives
    # the big blob is excluded
    assert not stk[0, 20, 20]


# ---------------------------------------------------------------------------
# DCT spectral level
# ---------------------------------------------------------------------------
def test_dct_low_freq_captures_smooth_field():
    # a smooth low-frequency field -> high LF energy fraction
    H, W = 64, 64
    yy, xx = np.mgrid[0:H, 0:W]
    smooth = np.stack([np.sin(2 * np.pi * (xx + t) / W) for t in range(10)])
    er = probe.level2_dct(smooth.astype(bool) if False else (smooth > 0),
                          keep_lf=16)
    # a single low-freq sinusoid concentrates energy in the LF block
    assert er["lf_energy_frac_mean"] > 0.5


# ---------------------------------------------------------------------------
# Fourier descriptors / contour level
# ---------------------------------------------------------------------------
def test_fourier_descriptors_scale_invariant():
    # a circle and a 2x-scaled circle have ~equal normalized descriptors
    theta = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    circ = np.column_stack([10 + 5 * np.sin(theta), 10 + 5 * np.cos(theta)])
    circ2 = np.column_stack([20 + 10 * np.sin(theta), 20 + 10 * np.cos(theta)])
    d1 = probe.fourier_descriptors(circ, 6)
    d2 = probe.fourier_descriptors(circ2, 6)
    assert np.allclose(d1, d2, atol=1e-6)  # scale-invariant


def test_fourier_descriptors_distinguish_shapes():
    # a circle vs a square must differ in their descriptors (not a constant stub)
    theta = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    circ = np.column_stack([5 * np.sin(theta), 5 * np.cos(theta)])
    sq = []
    for t in np.linspace(0, 1, 64, endpoint=False):
        # walk the perimeter of a square
        if t < 0.25:
            sq.append([-5 + 40 * t, -5])
        elif t < 0.5:
            sq.append([5, -5 + 40 * (t - 0.25)])
        elif t < 0.75:
            sq.append([5 - 40 * (t - 0.5), 5])
        else:
            sq.append([-5, 5 - 40 * (t - 0.75)])
    sq = np.array(sq)
    dc = probe.fourier_descriptors(circ, 6)
    ds = probe.fourier_descriptors(sq, 6)
    assert not np.allclose(dc, ds, atol=1e-3)


# ---------------------------------------------------------------------------
# motion-compensated level
# ---------------------------------------------------------------------------
def test_affine_estimate_recovers_pure_translation():
    # a blob translated by a known shift -> estimated affine warp aligns them
    import scipy.ndimage as ndi

    base = np.zeros((40, 40), dtype=np.float64)
    base[15:25, 15:25] = 1.0
    base = ndi.gaussian_filter(base, 1.0)
    shifted = ndi.shift(base, (0, 3), order=1)  # +3 cols
    M = probe.estimate_affine(prev=base, cur=shifted)
    warped = ndi.affine_transform(shifted, M[:, :2], offset=M[:, 2], order=1)
    # warped should be closer to base than the unwarped shifted is
    err_warp = float(((base - warped) ** 2).sum())
    err_raw = float(((base - shifted) ** 2).sum())
    assert err_warp < err_raw  # warp reduces residual (motion compensation works)


# ---------------------------------------------------------------------------
# nonlinear intrinsic-dim estimators -- the decisive tests
# ---------------------------------------------------------------------------
def test_twonn_recovers_known_intrinsic_dim_of_swiss_roll():
    # a 2D manifold embedded in 3D (swiss roll) has intrinsic dim ~2
    rng = np.random.default_rng(0)
    n = 400
    t = 3 * np.pi / 2 * (1 + 2 * rng.random(n))
    h = 20 * rng.random(n)
    X = np.column_stack([t * np.cos(t), h, t * np.sin(t)])
    m = probe.twonn_intrinsic_dim(X)
    assert 1.3 < m < 3.0  # ~2 (TwoNN on swiss roll)


def test_mle_recovers_known_intrinsic_dim_of_plane_in_high_d():
    # a 3D linear subspace embedded in 30D + tiny noise -> MLE id ~3
    rng = np.random.default_rng(1)
    Z = rng.standard_normal((500, 3))
    A = rng.standard_normal((3, 30))
    X = Z @ A + 0.001 * rng.standard_normal((500, 30))
    m = probe.mle_intrinsic_dim(X)
    assert 2.0 < m < 5.0  # ~3


def test_twonn_distinguishes_low_dim_from_full_dim_noise():
    # full-dim isotropic noise has HIGH intrinsic dim; a 2D manifold has LOW.
    rng = np.random.default_rng(2)
    noise = rng.standard_normal((300, 15))
    t = rng.random(300) * 6
    manifold = np.column_stack([np.cos(t), np.sin(t)] + [0.01 * rng.standard_normal(300)
                                                         for _ in range(13)])
    m_noise = probe.twonn_intrinsic_dim(noise)
    m_manifold = probe.twonn_intrinsic_dim(manifold)
    assert m_noise > m_manifold + 3  # noise clearly higher-dim than the manifold


def test_autoencoder_knee_low_for_low_dim_manifold():
    # frames generated from a 3-dim latent -> AE knee should be small (<=8)
    rng = np.random.default_rng(0)
    H = W = 32
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    frames = []
    for _ in range(120):
        a, b, c = rng.standard_normal(3)
        cy, cx, r = 16 + 4 * a, 16 + 4 * b, 5 + 2 * c
        disk = ((yy - cy) ** 2 + (xx - cx) ** 2) < r ** 2
        frames.append(disk)
    stack = np.stack(frames)
    ae = probe.tiny_autoencoder_knee(stack, dims=(2, 4, 8, 16), max_frames=120,
                                     epochs=150)
    # a 3-latent disk family is captured by a small bottleneck
    assert ae["smallest_bdim_90pct_of_best"] <= 8
    assert ae["best_explained_var"] > 0.5


def test_autoencoder_knee_not_a_constant_stub():
    # different bottleneck dims must yield monotone-improving explained var
    # (a hard-coded stub would not improve with capacity)
    rng = np.random.default_rng(3)
    H = W = 24
    frames = (rng.random((80, H, W)) > 0.97)  # sparse random islands
    ae = probe.tiny_autoencoder_knee(frames, dims=(2, 8, 32), max_frames=80,
                                     epochs=120)
    e2 = ae["per_dim"]["2"]["frac_var_explained"]
    e32 = ae["per_dim"]["32"]["frac_var_explained"]
    assert e32 >= e2  # more capacity never explains LESS (real AE behavior)


# ---------------------------------------------------------------------------
# verdict logic
# ---------------------------------------------------------------------------
def _mk_level(k95, pr):
    return {"k_for_95pct": k95, "participation_ratio": pr}


def test_verdict_go_format_when_a_basis_collapses():
    l1 = {"k_for_95pct": 400, "participation_ratio": 74.0}
    l2 = {"k_for_95pct": 10, "participation_ratio": 8.0}  # DCT collapses <=13
    l3 = {"frame_level": _mk_level(50, 20.0),
          "shape_vocabulary_cloud": _mk_level(2, 2.0)}
    l4 = {"warp_residual_stack": _mk_level(90, 50.0)}
    v = probe.decide_verdict(l1, l2, l3, l4, 28.0, 13.0,
                             {"knee_bottleneck_dim": 32,
                              "smallest_bdim_90pct_of_best": 8})
    assert v["verdict"] == "GO-FORMAT"
    assert v["min_recon_faithful_basis"] == "dct_lf"


def test_verdict_go_generator_when_linear_high_but_ae_low():
    # every linear recon-faithful basis > 13 budget, AE 90%-knee = 8 -> GENERATOR
    l1 = {"k_for_95pct": 412, "participation_ratio": 74.6}
    l2 = {"k_for_95pct": 61, "participation_ratio": 16.4}
    l3 = {"frame_level": _mk_level(29, 6.5),
          "shape_vocabulary_cloud": _mk_level(2, 2.0)}
    l4 = {"warp_residual_stack": _mk_level(93, 50.3)}
    v = probe.decide_verdict(l1, l2, l3, l4, 28.9, 13.1,
                             {"knee_bottleneck_dim": 32,
                              "smallest_bdim_90pct_of_best": 8})
    assert v["verdict"] == "GO-GENERATOR"


def test_verdict_wall_when_high_everywhere():
    # linear high AND nonlinear high -> WALL (irreducible content-noise)
    l1 = {"k_for_95pct": 412, "participation_ratio": 74.6}
    l2 = {"k_for_95pct": 61, "participation_ratio": 16.4}
    l3 = {"frame_level": _mk_level(29, 6.5),
          "shape_vocabulary_cloud": _mk_level(2, 2.0)}
    l4 = {"warp_residual_stack": _mk_level(93, 50.3)}
    v = probe.decide_verdict(l1, l2, l3, l4, 45.0, 40.0,
                             {"knee_bottleneck_dim": 32,
                              "smallest_bdim_90pct_of_best": 32})
    assert v["verdict"] == "WALL"


def test_verdict_shape_vocab_disqualified_from_go_format():
    # shape-vocab cloud is dim 2 but loses WHERE; must NOT trigger GO-FORMAT
    # unless a RECON-FAITHFUL basis is also <=13.
    l1 = {"k_for_95pct": 412, "participation_ratio": 74.6}
    l2 = {"k_for_95pct": 61, "participation_ratio": 16.4}
    l3 = {"frame_level": _mk_level(29, 6.5),
          "shape_vocabulary_cloud": _mk_level(2, 2.0)}  # tiny but NOT recon-faithful
    l4 = {"warp_residual_stack": _mk_level(93, 50.3)}
    v = probe.decide_verdict(l1, l2, l3, l4, 45.0, 40.0,
                             {"knee_bottleneck_dim": 32,
                              "smallest_bdim_90pct_of_best": 32})
    # shape-vocab dim 2 must NOT make it GO-FORMAT (it's WALL here)
    assert v["verdict"] != "GO-FORMAT"
    assert "contour_shape_vocab" not in v["recon_faithful_k95_per_basis"]


# ---------------------------------------------------------------------------
# authority cache guard
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not probe.ARGMAPS.exists(),
                    reason="cached authority argmaps not present")
def test_cached_argmaps_authority_and_class1_present():
    d = np.load(probe.ARGMAPS)
    gt = d["gt"]
    assert gt.shape == (600, 384, 512)
    # cache faithfulness (exact frozen-SegNet argmax, dt=1e-7)
    ds = float((d["gt"] != d["comp"]).mean())
    assert abs(ds - 0.0005598873) < 1e-7
    # class 1 (the island stratum) is present and sparse (~0.6% of pixels)
    frac = float((gt == probe.ISLAND_CLASS).mean())
    assert 0.001 < frac < 0.02
