# SPDX-License-Identifier: MIT
"""NO-FAKE behavioral tests for ``tac.optimization.aurora_mlx``.

These tests verify ACTUAL Aurora behavior on real arrays, not constants. Per
CLAUDE.md "NO FAKE IMPLEMENTATIONS" Class 2: if the function body were replaced
by ``return canonical_markers`` (or by plain Muon polar for the tall case),
these tests MUST fail. The headline behavioral guards are:

- ``test_tall_leverage_uniformity_beats_plain_polar`` — the core Aurora claim:
  for a tall matrix, Aurora drives row-norm-squared toward the uniform target
  ``n/m`` with MUCH smaller spread than plain Muon polar. FAILS if the body
  reverts to plain polar.
- ``test_square_reduces_to_muon_polar`` — Aurora == Muon for square inputs.
- ``test_parity_*_vs_pytorch_reference`` — numerical parity against the cloned
  upstream PyTorch ``aurora()`` / ``polar()`` reference (skipped if the clone is
  not present on disk).

MLX is required (Apple Silicon); the module is a ``[macOS-MLX research-signal]``
kernel and asserts no contest score.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from tac.optimization.aurora_mlx import (  # noqa: E402
    AURORA_DEFAULT_PP_BETA,
    AURORA_DEFAULT_PP_ITERATIONS,
    AURORA_SIMPLE_QUINTIC_COEFFS,
    AURORA_SIMPLE_QUINTIC_STEPS,
    AURORA_SOURCE_LICENSE,
    AURORA_SOURCE_REPO,
    AuroraMlxError,
    aurora_leverage_uniform_polar_mlx,
    aurora_simple_quintic_polar_mlx,
    aurora_update_mlx,
    classify_matrix_shape,
)

# Location of the cloned upstream reference (SSD scratch; optional).
_REF_SRC = "/Volumes/VertigoDataTier/pact/tilde_scratch/aurora-release/src"


def _reference_available() -> bool:
    return os.path.isdir(_REF_SRC) and os.path.isfile(
        os.path.join(_REF_SRC, "aurora.py")
    )


def _load_reference():
    if _REF_SRC not in sys.path:
        sys.path.insert(0, _REF_SRC)
    import aurora as _aurora_mod  # type: ignore[import-not-found]
    import polar as _polar_mod  # type: ignore[import-not-found]

    return _aurora_mod.aurora, _polar_mod.polar


# ---------------------------------------------------------------------------
# Provenance constants (these are metadata, not behavior — kept minimal).
# ---------------------------------------------------------------------------


def test_provenance_constants_are_the_verified_aurora_values():
    # Verified against the cloned repo 2026-06-09.
    assert AURORA_SOURCE_REPO == "https://github.com/tilde-research/aurora-release"
    assert AURORA_SOURCE_LICENSE == "MIT"
    # Aurora's polar.py uses the 12-step simple-quintic, NOT Keller-Jordan 5-step.
    assert AURORA_SIMPLE_QUINTIC_COEFFS == (2.0, -1.5, 0.5)
    assert AURORA_SIMPLE_QUINTIC_STEPS == 12
    assert AURORA_DEFAULT_PP_ITERATIONS == 2
    assert AURORA_DEFAULT_PP_BETA == 0.5


# ---------------------------------------------------------------------------
# Shape classification
# ---------------------------------------------------------------------------


def test_classify_matrix_shape():
    assert classify_matrix_shape(64, 8) == "tall"
    assert classify_matrix_shape(8, 64) == "wide"
    assert classify_matrix_shape(16, 16) == "square"


def test_classify_matrix_shape_rejects_nonpositive():
    with pytest.raises(AuroraMlxError):
        classify_matrix_shape(0, 8)
    with pytest.raises(AuroraMlxError):
        classify_matrix_shape(8, -1)


# ---------------------------------------------------------------------------
# Polar kernel behavior
# ---------------------------------------------------------------------------


def test_polar_maps_singular_values_to_one():
    # A well-conditioned square matrix: polar should give an orthonormal matrix
    # (all singular values driven to 1 => U^T U ~= I).
    mx.random.seed(0)
    a = mx.random.normal((24, 24))
    p = aurora_simple_quintic_polar_mlx(a, cast_float32_to_bfloat16=False)
    gram = p.T @ p
    eye = mx.eye(24)
    max_off = float(mx.max(mx.abs(gram - eye)).item())
    # bf16-tuned quintic converges to ~1e-2; we run f32 here so tighter.
    assert max_off < 5e-2, max_off


def test_polar_tall_returns_same_shape():
    mx.random.seed(1)
    a = mx.random.normal((40, 6))
    p = aurora_simple_quintic_polar_mlx(a, cast_float32_to_bfloat16=False)
    assert tuple(p.shape) == (40, 6)


def test_polar_rejects_non_2d():
    with pytest.raises(AuroraMlxError):
        aurora_simple_quintic_polar_mlx(mx.zeros((4, 4, 4)))


def test_polar_rejects_bad_steps():
    with pytest.raises(AuroraMlxError):
        aurora_simple_quintic_polar_mlx(mx.zeros((4, 4)), steps=0)


# ---------------------------------------------------------------------------
# CORE behavioral guard: square reduces to Muon polar
# ---------------------------------------------------------------------------


def test_square_reduces_to_muon_polar():
    """Aurora's leverage-uniform polar == plain polar for square matrices.

    This is the documented "reduces to standard Muon" property. If the body
    instead always ran the tall preconditioning loop, this would NOT be exact.
    """
    mx.random.seed(2)
    a = mx.random.normal((20, 20))
    lev = aurora_leverage_uniform_polar_mlx(a, apply_aspect_scale=False)
    plain = aurora_simple_quintic_polar_mlx(a, cast_float32_to_bfloat16=False)
    max_diff = float(mx.max(mx.abs(lev - plain)).item())
    assert max_diff == 0.0, max_diff


def test_square_aspect_scale_is_one():
    # For square m==n, max(1, sqrt(m/n)) == 1, so scaled == unscaled.
    mx.random.seed(3)
    a = mx.random.normal((12, 12))
    unscaled = aurora_leverage_uniform_polar_mlx(a, apply_aspect_scale=False)
    scaled = aurora_leverage_uniform_polar_mlx(a, apply_aspect_scale=True)
    assert float(mx.max(mx.abs(unscaled - scaled)).item()) == 0.0


# ---------------------------------------------------------------------------
# CORE behavioral guard: tall leverage uniformity (THE Aurora mechanism)
# ---------------------------------------------------------------------------


def test_tall_leverage_uniformity_beats_plain_polar():
    """The headline Aurora claim, made falsifiable.

    For a tall (m>n) matrix, Aurora drives the per-row ||U_i||^2 toward the
    uniform target n/m with FAR smaller spread than plain Muon polar (which
    inherits non-uniform leverage). If this function body reverted to plain
    polar, the std reduction would vanish and this test would FAIL.
    """
    mx.random.seed(4)
    m, n = 64, 8
    g = mx.random.normal((m, n))
    target = n / m

    u_aurora = aurora_leverage_uniform_polar_mlx(
        g, pp_iterations=2, apply_aspect_scale=False
    )
    u_muon = aurora_simple_quintic_polar_mlx(g, cast_float32_to_bfloat16=False)

    rows_aurora = mx.sum(u_aurora.astype(mx.float32) ** 2, axis=-1)
    rows_muon = mx.sum(u_muon.astype(mx.float32) ** 2, axis=-1)

    mean_aurora = float(mx.mean(rows_aurora).item())
    std_aurora = float(mx.std(rows_aurora).item())
    std_muon = float(mx.std(rows_muon).item())

    # Aurora mean row_sq close to target n/m.
    assert abs(mean_aurora - target) < 0.02, (mean_aurora, target)
    # Aurora leverage spread MUCH smaller than Muon's (the whole point).
    assert std_aurora < 0.5 * std_muon, (std_aurora, std_muon)
    # And concretely small in absolute terms after 2 iterations.
    assert std_aurora < 0.02, std_aurora


def test_more_pp_iterations_tighten_leverage():
    """More preconditioning iterations => tighter row-norm uniformity."""
    mx.random.seed(5)
    m, n = 96, 6
    g = mx.random.normal((m, n))
    target = n / m

    def spread(pp):
        u = aurora_leverage_uniform_polar_mlx(
            g, pp_iterations=pp, apply_aspect_scale=False
        )
        rows = mx.sum(u.astype(mx.float32) ** 2, axis=-1)
        return float(mx.max(mx.abs(rows - target)).item())

    s1 = spread(1)
    s3 = spread(3)
    # pp=1 is a single normalize+polar (no correction); pp=3 should be tighter.
    assert s3 < s1, (s1, s3)


def test_wide_transposes_and_orthogonalizes():
    """Wide (m<n) is handled via transpose-to-tall; result stays finite & shaped."""
    mx.random.seed(6)
    m, n = 8, 64
    g = mx.random.normal((m, n))
    u = aurora_leverage_uniform_polar_mlx(g, apply_aspect_scale=False)
    assert tuple(u.shape) == (m, n)
    assert bool(mx.all(mx.isfinite(u)).item())
    # For wide, U U^T ~= I_m (row-orthonormal on the short dim).
    gram = u.astype(mx.float32) @ u.astype(mx.float32).T
    eye = mx.eye(m)
    assert float(mx.max(mx.abs(gram - eye)).item()) < 1e-1


# ---------------------------------------------------------------------------
# N-D (conv) reshape handling
# ---------------------------------------------------------------------------


def test_conv4d_reshape_roundtrip_shape():
    # (out, kh, kw, in) MLX conv layout -> flatten (out, kh*kw*in) -> reshape back
    mx.random.seed(7)
    w = mx.random.normal((72, 3, 3, 18))  # like blocks.5.conv.weight
    u = aurora_leverage_uniform_polar_mlx(w, apply_aspect_scale=True)
    assert tuple(u.shape) == (72, 3, 3, 18)
    assert bool(mx.all(mx.isfinite(u)).item())


def test_conv4d_matches_manual_flatten():
    # The N-D path must equal: reshape -> 2d kernel -> reshape.
    mx.random.seed(8)
    w = mx.random.normal((24, 3, 3, 8))
    nd = aurora_leverage_uniform_polar_mlx(w, apply_aspect_scale=True)
    rows = 24
    cols = 3 * 3 * 8
    flat = mx.reshape(w, (rows, cols))
    twod = aurora_leverage_uniform_polar_mlx(flat, apply_aspect_scale=True)
    manual = mx.reshape(twod, (24, 3, 3, 8))
    assert float(mx.max(mx.abs(nd - manual)).item()) == 0.0


def test_leverage_uniform_rejects_ndim1():
    with pytest.raises(AuroraMlxError):
        aurora_leverage_uniform_polar_mlx(mx.zeros((10,)))


def test_leverage_uniform_rejects_bad_pp():
    with pytest.raises(AuroraMlxError):
        aurora_leverage_uniform_polar_mlx(mx.zeros((8, 4)), pp_iterations=0)
    with pytest.raises(AuroraMlxError):
        aurora_leverage_uniform_polar_mlx(mx.zeros((8, 4)), pp_beta=0.0)


# ---------------------------------------------------------------------------
# Full update: momentum, weight decay, scaling
# ---------------------------------------------------------------------------


def test_aurora_update_applies_weight_decay():
    """With zero gradient, only decoupled weight decay should shrink W."""
    w = mx.ones((8, 8))
    g = mx.zeros((8, 8))
    mom = mx.zeros((8, 8))
    new_w, new_mom = aurora_update_mlx(
        w, g, mom, eta=0.1, weight_decay=0.5, mu=0.95, nesterov=True
    )
    # update from zero grad/momentum -> polar(0)=0, so W *= (1 - 0.1*0.5) = 0.95
    expected = 1.0 - 0.1 * 0.5
    assert abs(float(mx.mean(new_w).item()) - expected) < 1e-6
    # momentum stays zero (lerp of zeros).
    assert float(mx.max(mx.abs(new_mom)).item()) == 0.0


def test_aurora_update_momentum_accumulates():
    mx.random.seed(9)
    w = mx.zeros((8, 8))
    g = mx.random.normal((8, 8))
    mom = mx.zeros((8, 8))
    _, new_mom = aurora_update_mlx(w, g, mom, mu=0.9, nesterov=False)
    # new_mom = mom + (1-mu)*(g - mom) = 0.1 * g
    expected = 0.1 * g
    assert float(mx.max(mx.abs(new_mom - expected)).item()) < 1e-6


def test_aurora_update_changes_weight():
    mx.random.seed(10)
    w = mx.random.normal((16, 4))
    g = mx.random.normal((16, 4))
    mom = mx.zeros((16, 4))
    new_w, _ = aurora_update_mlx(w, g, mom, eta=0.05, weight_decay=0.0)
    # Non-trivial step occurred.
    assert float(mx.max(mx.abs(new_w - w)).item()) > 1e-4


def test_aurora_update_rejects_shape_mismatch():
    with pytest.raises(AuroraMlxError):
        aurora_update_mlx(mx.zeros((8, 4)), mx.zeros((8, 5)), mx.zeros((8, 4)))
    with pytest.raises(AuroraMlxError):
        aurora_update_mlx(mx.zeros((8, 4)), mx.zeros((8, 4)), mx.zeros((4, 4)))
    with pytest.raises(AuroraMlxError):
        aurora_update_mlx(mx.zeros((8,)), mx.zeros((8,)), mx.zeros((8,)))


def test_aurora_update_rejects_bad_hparams():
    w = mx.zeros((4, 4))
    with pytest.raises(AuroraMlxError):
        aurora_update_mlx(w, w, w, mu=0.0)
    with pytest.raises(AuroraMlxError):
        aurora_update_mlx(w, w, w, eta=0.0)
    with pytest.raises(AuroraMlxError):
        aurora_update_mlx(w, w, w, weight_decay=-0.1)


# ---------------------------------------------------------------------------
# Numerical parity vs the cloned upstream PyTorch reference (NO-FAKE anchor)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _reference_available(),
    reason="cloned aurora-release reference not present on SSD scratch",
)
def test_parity_polar_vs_pytorch_reference():
    """MLX polar (bf16) matches the cloned torch polar (bf16) to bf16 precision.

    Both run the identical 12-step simple-quintic in bfloat16; the only possible
    divergence is matmul reduction-ORDER (a few bf16 ULPs on some shapes). We
    assert: (a) the gap is at most a few bf16 ULPs, AND (b) BOTH outputs are
    equally-valid polar factors (identical ``U^T U - I`` error) — proving the
    gap is precision, not logic. A real logic error (wrong coeffs, wrong steps,
    missing transpose) would blow either check up by orders of magnitude.
    """
    torch = pytest.importorskip("torch")
    _, torch_polar = _load_reference()
    rng = np.random.default_rng(11)
    for m, n in [(64, 8), (16, 16), (8, 64), (40, 6), (20, 20)]:
        a = rng.standard_normal((m, n)).astype(np.float32)
        ref = torch_polar(torch.tensor(a)).float().numpy()
        got = np.asarray(
            aurora_simple_quintic_polar_mlx(
                mx.array(a), cast_float32_to_bfloat16=True
            ).astype(mx.float32)
        )
        # A few bf16 ULPs at O(1) magnitude.
        max_diff = float(np.abs(got - ref).max())
        assert max_diff < 1e-2, (m, n, max_diff)
        # Both are equally-valid polar factors: identical orthonormality error.
        # (np.errstate suppresses the spurious macOS Accelerate BLAS warnings
        # for these small matmuls; outputs are finite.)
        k = min(m, n)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            if m >= n:
                ref_orth = float(np.abs(ref.T @ ref - np.eye(n)).max())
                got_orth = float(np.abs(got.T @ got - np.eye(n)).max())
            else:
                ref_orth = float(np.abs(ref @ ref.T - np.eye(m)).max())
                got_orth = float(np.abs(got @ got.T - np.eye(m)).max())
        assert np.isfinite(ref_orth) and np.isfinite(got_orth), (m, n)
        assert abs(ref_orth - got_orth) < 1e-3, (m, n, ref_orth, got_orth, k)


@pytest.mark.skipif(
    not _reference_available(),
    reason="cloned aurora-release reference not present on SSD scratch",
)
def test_polar_logic_exact_in_float64_vs_numpy_reference():
    """Pin the polar ALGORITHM exactly (no bf16 noise) against a numpy mirror.

    Re-implements the 12-step simple-quintic in numpy float64 and asserts the
    MLX float32 polar matches it tightly. This proves the iteration logic
    (coefficients, step count, transpose, normalization) is exactly right,
    independent of bf16 reduction order.
    """
    rng = np.random.default_rng(21)

    def numpy_polar(g: np.ndarray) -> np.ndarray:
        x = g.astype(np.float64)
        transposed = x.shape[-2] > x.shape[-1]
        if transposed:
            x = x.T
        x = x / (np.linalg.norm(x) + 1e-7)
        a, b, c = AURORA_SIMPLE_QUINTIC_COEFFS
        # macOS Accelerate BLAS emits spurious divide/overflow RuntimeWarnings
        # for some small matmul sizes even when the result is finite and
        # correct (verified: spectral norm stays ~1.0 and output is finite).
        # Suppress the spurious flags; the finite-output assertion below is the
        # real divergence guard.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            for _ in range(AURORA_SIMPLE_QUINTIC_STEPS):
                aa = x @ x.T
                bb = b * aa + c * (aa @ aa)
                x = a * x + bb @ x
        out = x.T if transposed else x
        assert np.isfinite(out).all(), "numpy reference polar diverged"
        return out

    for m, n in [(64, 8), (16, 16), (8, 64), (40, 6)]:
        a = rng.standard_normal((m, n)).astype(np.float32)
        ref = numpy_polar(a)
        got = np.asarray(
            aurora_simple_quintic_polar_mlx(
                mx.array(a), cast_float32_to_bfloat16=False
            ).astype(mx.float32)
        )
        max_diff = float(np.abs(got - ref).max())
        # f32 vs f64 of an identical 12-step iteration: tight (~1e-3). A logic
        # error (wrong coeffs/steps/transpose) would be orders of magnitude
        # larger, not sub-1e-3.
        assert max_diff < 2e-3, (m, n, max_diff)


@pytest.mark.skipif(
    not _reference_available(),
    reason="cloned aurora-release reference not present on SSD scratch",
)
def test_parity_square_update_vs_pytorch_reference():
    """Square update has no f32 preconditioning loop => near-exact parity."""
    torch = pytest.importorskip("torch")
    torch_aurora, _ = _load_reference()
    rng = np.random.default_rng(12)
    m = n = 16
    w_np = rng.standard_normal((m, n)).astype(np.float32)
    g_np = rng.standard_normal((m, n)).astype(np.float32)
    mom_np = np.zeros((m, n), dtype=np.float32)

    wt = torch.tensor(w_np.copy())
    torch_aurora(
        wt,
        torch.tensor(g_np.copy()),
        torch.tensor(mom_np.copy()),
        eta=0.05,
        weight_decay=0.025,
        mu=0.95,
        nesterov=True,
        pp_iterations=2,
        pp_beta=0.5,
    )
    wm, _ = aurora_update_mlx(
        mx.array(w_np.copy()),
        mx.array(g_np.copy()),
        mx.array(mom_np.copy()),
        eta=0.05,
        weight_decay=0.025,
        mu=0.95,
        nesterov=True,
        pp_iterations=2,
        pp_beta=0.5,
        polar_cast_float32_to_bfloat16=True,
    )
    max_diff = float(np.abs(np.asarray(wm) - wt.numpy()).max())
    assert max_diff < 1e-6, max_diff


@pytest.mark.skipif(
    not _reference_available(),
    reason="cloned aurora-release reference not present on SSD scratch",
)
def test_parity_tall_update_vs_pytorch_reference():
    """Tall update: f32 preconditioning loop differs only by reduction order.

    Parity should be tight (<2e-3 absolute on ~O(1) weights); a logic error
    (e.g. wrong target n/m, missing transpose) would blow this up by orders of
    magnitude.
    """
    torch = pytest.importorskip("torch")
    torch_aurora, _ = _load_reference()
    rng = np.random.default_rng(13)
    for m, n in [(64, 8), (8, 64)]:
        w_np = rng.standard_normal((m, n)).astype(np.float32)
        g_np = rng.standard_normal((m, n)).astype(np.float32)
        mom_np = np.zeros((m, n), dtype=np.float32)

        wt = torch.tensor(w_np.copy())
        torch_aurora(
            wt,
            torch.tensor(g_np.copy()),
            torch.tensor(mom_np.copy()),
            eta=0.05,
            weight_decay=0.025,
            mu=0.95,
            nesterov=True,
            pp_iterations=2,
            pp_beta=0.5,
        )
        wm, _ = aurora_update_mlx(
            mx.array(w_np.copy()),
            mx.array(g_np.copy()),
            mx.array(mom_np.copy()),
            eta=0.05,
            weight_decay=0.025,
            mu=0.95,
            nesterov=True,
            pp_iterations=2,
            pp_beta=0.5,
            polar_cast_float32_to_bfloat16=True,
        )
        max_diff = float(np.abs(np.asarray(wm) - wt.numpy()).max())
        assert max_diff < 2e-3, (m, n, max_diff)


# ---------------------------------------------------------------------------
# Regime-fit guard: HNeRV Muon-eligible weights are all WIDE (predicted-null)
# ---------------------------------------------------------------------------


def test_hnerv_muon_eligible_partition_has_no_tall_matrices():
    """Document & guard the decisive regime-fit fact for the contest decoder.

    Aurora helps TALL matrices. The HNeRV Muon-eligible partition is empirically
    all WIDE (0 tall, 0 square) => Aurora's benefit is predicted ~0. If a future
    architecture change introduces tall Muon-eligible weights, THIS test breaks
    and signals "re-evaluate Aurora's reactivation criterion".
    """
    pytest.importorskip("mlx.nn")
    from mlx.utils import tree_flatten

    from tac.local_acceleration.pr95_hnerv_mlx import (
        HNeRVDecoderMLX,
        partition_pr95_mlx_parameter_names,
    )

    dec = HNeRVDecoderMLX(latent_dim=28, base_channels=36)
    split = partition_pr95_mlx_parameter_names(dec.parameters())
    flat = dict(tree_flatten(dec.parameters()))

    classes = {"tall": 0, "wide": 0, "square": 0}
    for name in split["muon"]:
        shp = tuple(int(x) for x in flat[name].shape)
        if len(shp) == 4:
            rows, cols = shp[0], math.prod(shp[1:])
        else:
            rows, cols = shp[0], shp[1]
        classes[classify_matrix_shape(rows, cols)] += 1

    assert classes["muon" if False else "tall"] == 0, classes
    assert classes["square"] == 0, classes
    assert classes["wide"] >= 1, classes
    # The genuinely tall weight (stem) is in the AdamW partition, not Muon.
    assert any("stem" in n for n in split["adamw"])
