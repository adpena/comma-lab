"""DM1 minimal-cure math tests (Stiefel-W + code spectral-entropy).

NO-FAKE self-review pass-1 for the byte-free DM1 cure (design memo
``per_stage_fractal_optimizer_priming_reheat_anneal_design_20260629.md`` §0):

  1. the Stiefel projection ACTUALLY orthonormalizes (``‖WᵀW−I‖_F`` small
     post-projection; NOT a no-op, NOT the Muon ``newton_schulz5`` which leaves
     residual ~1.3);
  2. the load-bearing identity ``WᵀW=I ⟹ PR(M)=PR(cov(code))`` (on an
     orthonormalized W + random code), and that it BREAKS for a non-orthonormal W
     (so the cure is load-bearing, not vacuous);
  3. the code spectral-entropy penalty computes on REAL code stats and has a
     NONZERO gradient (perturbing code changes the penalty);
  4. numpy-reference parity for the MLX twin (one math, two backends);
  5. Stiefel projection NEUTRALIZES a uniform column rescale (the WD=0-on-W
     intent: AdamW's W(1−lr·wd) uniform shrink is undone by column re-normalization).

These are advisory MLX/numpy math locks; they make NO contest-score claim.
"""
import numpy as np
import pytest

from tac.boundary_math.lever_b_levelset_generator import (
    code_spectral_entropy_penalty,
    code_spectrum_participation_ratio,
    film_modulation_participation_ratio,
)

mx = pytest.importorskip("mlx.core")
from tac.optimization.md_decoupling import (  # noqa: E402
    newton_schulz5,
    stiefel_project_columns,
    stiefel_residual,
)

# film.weight shape for the BASELINE witness: (2*hidden_dim*n_hidden, mod_dim)
_OUT, _MOD = 2 * 96 * 4, 32


# --- (1) the Stiefel projection ACTUALLY orthonormalizes (not a no-op) ---
def test_stiefel_projection_orthonormalizes_columns():
    rng = np.random.default_rng(0)
    W = mx.array((rng.standard_normal((_OUT, _MOD)) * 0.3).astype(np.float32))
    res_before = stiefel_residual(W)
    Wo = stiefel_project_columns(W)
    res_after = stiefel_residual(Wo)
    # a RANDOM W is far from Stiefel; the projection must drive the residual to the fp32 floor.
    assert res_before > 0.5, f"test fixture not far-from-Stiefel: {res_before}"
    assert res_after < 0.05, f"projection did not orthonormalize: ‖WᵀW−I‖_F={res_after}"
    # column norms ~ 1 (orthonormal columns)
    colnorm = mx.sqrt(mx.sum(Wo * Wo, axis=0))
    assert float(mx.max(mx.abs(colnorm - 1.0))) < 0.05


def test_stiefel_projection_is_not_a_noop_vs_input():
    rng = np.random.default_rng(1)
    W = mx.array((rng.standard_normal((_OUT, _MOD)) * 1.7).astype(np.float32))
    Wo = stiefel_project_columns(W)
    # the output must DIFFER from the input (proves work happened, not an identity passthrough)
    assert float(mx.max(mx.abs(Wo - W))) > 1e-3


def test_stiefel_project_rejects_non_2d():
    with pytest.raises(ValueError):
        stiefel_project_columns(mx.zeros((5,)))


# --- (review R1-L3/R3) tall-matrix guard: a WIDE matrix cannot have orthonormal columns ---
def test_stiefel_project_rejects_wide_matrix():
    # out < in => WᵀW (in x in) is rank-deficient => WᵀW=I_in unreachable; must fail closed (else a
    # silent FAKE projection whose residual never reaches the floor).
    wide = mx.array(np.random.default_rng(11).standard_normal((8, 32)).astype(np.float32))
    with pytest.raises(ValueError):
        stiefel_project_columns(wide)
    # the BASELINE film.weight (tall: 768 x 32) must be accepted.
    tall = mx.array(np.random.default_rng(12).standard_normal((_OUT, _MOD)).astype(np.float32))
    _ = stiefel_project_columns(tall)  # no raise


# --- (review R1-L4/R3-L3) the PR identity holds through the REAL cubic projection, not just QR ---
def test_pr_identity_holds_through_cubic_projection_at_realistic_residual():
    # test_stiefel_identity_pr_M_equals_pr_cov_code uses a PERFECT (QR) orthonormal W. The SHIPPED
    # cure uses stiefel_project_columns (cubic Newton-Schulz, ~1e-2 residual). Compose the ACTUAL
    # projection end-to-end and assert PR(M) ~ PR(cov(code)) at that realistic (loose) tolerance.
    rng = np.random.default_rng(21)
    S = 200
    code_np = rng.standard_normal((S, _MOD)).astype(np.float32)
    W_raw = mx.array((rng.standard_normal((_OUT, _MOD)) * 0.7).astype(np.float32))
    W = stiefel_project_columns(W_raw)
    res = stiefel_residual(W)
    assert res < 0.05, f"projection did not reach the fp32 floor: {res}"
    M = np.asarray(mx.array(code_np) @ W.T, np.float32)        # modulation through the SHIPPED W
    pr_M = film_modulation_participation_ratio(M)
    pr_c = code_spectrum_participation_ratio(code_np)
    # the ~1e-2 residual perturbs the spectrum slightly => a LOOSE (not 1e-6) tolerance, but the
    # identity must clearly hold (the cure is real, not vacuous).
    assert pr_M == pytest.approx(pr_c, rel=0.05, abs=0.2), f"PR(M)={pr_M} vs PR(cov code)={pr_c}"


# --- (review R3-L1) NS5 regression lock: the Muon newton_schulz5 is NOT a Stiefel projection ---
def test_newton_schulz5_is_not_a_stiefel_projection():
    # The cure deliberately uses the CUBIC polar iteration, NOT the Muon QUINTIC newton_schulz5 (which
    # orthogonalizes a DIRECTION, leaving column residual ~1.3). Lock the distinction so a future
    # "just reuse newton_schulz5" refactor cannot silently turn the cure into a FAKE projection.
    rng = np.random.default_rng(31)
    W = mx.array((rng.standard_normal((_OUT, _MOD)) * 0.3).astype(np.float32))
    ns5 = newton_schulz5(W)
    assert stiefel_residual(ns5) > 0.5, (
        f"newton_schulz5 unexpectedly orthonormalized columns (residual {stiefel_residual(ns5)}); the "
        "cubic-vs-quintic distinction has rotted -- the cure must use stiefel_project_columns")
    # and the cubic projection on the SAME input DOES reach the floor (the contrast that matters).
    assert stiefel_residual(stiefel_project_columns(W)) < 0.05


# --- (2) the load-bearing identity PR(M) = PR(cov(code)) under WᵀW=I ---
def test_stiefel_identity_pr_M_equals_pr_cov_code():
    rng = np.random.default_rng(2)
    S = 200
    code = rng.standard_normal((S, _MOD)).astype(np.float64)
    # orthonormal-COLUMNS W via reduced QR (exact, numpy-portable) — WᵀW = I_mod
    W = np.linalg.qr(rng.standard_normal((_OUT, _MOD)))[0]
    # numpy 1.26.4 + Apple Accelerate spuriously raises FP-exception RuntimeWarnings on matmul even
    # for fully-finite inputs (values verified correct: WᵀW off-diag ~1e-15, M finite); silence them.
    with np.errstate(all="ignore"):
        assert np.allclose(W.T @ W, np.eye(_MOD), atol=1e-8)
        M = code @ W.T                  # (S, out) per-pair modulation = code @ Wᵀ
    pr_M = film_modulation_participation_ratio(M)
    pr_c = code_spectrum_participation_ratio(code)
    # the isometry preserves the nonzero spectrum => PR(M) == PR(cov(code)) (numerically exact)
    assert pr_M == pytest.approx(pr_c, rel=1e-6, abs=1e-6)


def test_identity_breaks_for_non_orthonormal_w():
    rng = np.random.default_rng(3)
    S = 200
    code = rng.standard_normal((S, _MOD)).astype(np.float64)
    # a deliberately ill-conditioned (NON-orthonormal) W concentrates the modulation spectrum
    raw = rng.standard_normal((_OUT, _MOD))
    raw[:, 0] *= 12.0                   # blow up one column => W shapes the resonance
    with np.errstate(all="ignore"):    # spurious Accelerate matmul warnings (see above)
        M = code @ raw.T
    pr_M = film_modulation_participation_ratio(M)
    pr_c = code_spectrum_participation_ratio(code)
    # the identity is LOAD-BEARING: without WᵀW=I, PR(M) collapses well below PR(cov(code))
    assert pr_M < 0.7 * pr_c


# --- (3) the code spectral-entropy penalty: real stats + nonzero gradient ---
def test_code_spectral_entropy_off_when_beta_nonpositive():
    rng = np.random.default_rng(4)
    code = rng.standard_normal((64, _MOD))
    assert code_spectral_entropy_penalty(code, 0.0) == 0.0
    assert code_spectral_entropy_penalty(code, -1.0) == 0.0


def test_code_spectral_entropy_depends_on_real_code_stats():
    rng = np.random.default_rng(5)
    # a uniform-spectrum code (high PR) vs a rank-1-ish code (low PR) => different penalty
    uniform = rng.standard_normal((200, _MOD))
    collapsed = np.zeros((200, _MOD))
    collapsed[:, 0] = rng.standard_normal(200)   # all energy in one direction => PR ~ 1
    p_uniform = code_spectral_entropy_penalty(uniform, beta=1.0)
    p_collapsed = code_spectral_entropy_penalty(collapsed, beta=1.0)
    # penalty = -log(PR); collapsed PR ~ 1 => -log ~ 0; uniform PR >> 1 => -log << 0
    assert p_collapsed > p_uniform
    assert code_spectrum_participation_ratio(uniform) > 5.0
    assert code_spectrum_participation_ratio(collapsed) < 1.5


def test_code_spectral_entropy_mlx_gradient_is_nonzero_and_raises_pr():
    # MLX twin of the trainer's inline penalty; prove a NONZERO gradient w.r.t. the code (NO-FAKE:
    # the penalty must actually push the code spectrum, not be a constant).
    rng = np.random.default_rng(6)
    code0 = mx.array(rng.standard_normal((128, _MOD)).astype(np.float32))

    def penalty(code, beta=1.0):
        Cc = code - mx.mean(code, axis=0, keepdims=True)
        Cov = Cc.T @ Cc
        ctr = mx.sum(Cc * Cc)
        cfro2 = mx.sum(Cov * Cov)
        cpr = (ctr * ctr) / (cfro2 + 1e-12)
        return -beta * mx.log(cpr + 1e-12)

    g = mx.grad(penalty)(code0)
    gnorm = float(mx.sqrt(mx.sum(g * g)))
    assert gnorm > 1e-6, "spectral-entropy penalty has a zero gradient (would be a no-op)"
    # one gradient-descent step on the penalty must RAISE PR(cov(code)) (the intent)
    pr_before = code_spectrum_participation_ratio(np.asarray(code0, np.float32))
    code1 = code0 - 0.05 * g
    pr_after = code_spectrum_participation_ratio(np.asarray(code1, np.float32))
    assert pr_after > pr_before


def test_mlx_penalty_matches_numpy_reference():
    # one math, two backends: the MLX inline expression == the numpy reference (fp32 tol)
    rng = np.random.default_rng(7)
    code_np = rng.standard_normal((200, _MOD)).astype(np.float32)
    code_mx = mx.array(code_np)
    Cc = code_mx - mx.mean(code_mx, axis=0, keepdims=True)
    Cov = Cc.T @ Cc
    cpr = (mx.sum(Cc * Cc) ** 2) / (mx.sum(Cov * Cov) + 1e-12)
    mlx_penalty = float(-1.0 * mx.log(cpr + 1e-12))
    np_penalty = code_spectral_entropy_penalty(code_np, beta=1.0)
    assert mlx_penalty == pytest.approx(np_penalty, rel=1e-3, abs=1e-3)


# --- (4b) byte-identity guard: both DM1 flags DEFAULT-OFF in the trainer argparse ---
def test_dm1_trainer_flags_default_off():
    # Default-off => the loss-term branch + the post-update Stiefel projection are skipped =>
    # the forward/loss/checkpoint path is bit-identical to the pre-DM1 trainer. A default flip would
    # silently break byte-identity, so guard it at the source.
    from pathlib import Path
    src = Path(__file__).resolve().parents[3] / (
        "experiments/train_levelset_witness_realized_through_R_mlx.py")
    text = src.read_text()
    # --film-stiefel is store_true (default False), NOT store_true-with-default-True / store_false.
    assert '"--film-stiefel", action="store_true"' in text
    assert '"--film-stiefel"' in text and "store_false" not in text.split('"--film-stiefel"')[1][:120]
    # --code-spectral-entropy-weight defaults to 0.0 (off).
    assert '"--code-spectral-entropy-weight", type=float, default=0.0' in text
    # the loss term + projection are GATED on the off-defaults.
    assert "if code_spec_w > 0.0 and code_spec_idx is not None:" in text
    assert "if args.film_stiefel:" in text
    # (review C1) --dm1-telemetry is store_true (default OFF) and the telemetry GATE is widened to
    # include it (so the A0 baseline can log PR without a DM1 lever) WITHOUT changing the default path.
    assert '"--dm1-telemetry", action="store_true"' in text
    assert "args.film_stiefel or code_spec_w > 0.0 or args.dm1_telemetry" in text
    # (review C2) --anneal-epochs defaults to None (=> use --epochs => bit-identical cosine denominator).
    assert '"--anneal-epochs", type=int, default=None' in text
    # the anneal denominator falls back to args.epochs when --anneal-epochs is unset (byte-identity).
    assert 'getattr(args, "anneal_epochs", None) or args.epochs' in text


# --- (5) WD neutralization: projection is invariant to a uniform column rescale ---
def test_stiefel_projection_neutralizes_uniform_rescale():
    rng = np.random.default_rng(8)
    W = mx.array((rng.standard_normal((_OUT, _MOD)) * 0.5).astype(np.float32))
    Wo = stiefel_project_columns(W)
    # AdamW decoupled WD scales W by a uniform scalar (1 - lr*wd); the projection must undo it.
    for alpha in (0.5, 0.9, 1.5):
        Wo_scaled = stiefel_project_columns(alpha * W)
        assert float(mx.max(mx.abs(Wo_scaled - Wo))) < 1e-4, (
            f"projection not scale-invariant at alpha={alpha} => WD not neutralized")
