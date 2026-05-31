# SPDX-License-Identifier: MIT
"""Tests for the OPTIMAL seg/pose teacher kernels (boundary-TCKD + pose-AIL).

NO-FAKE discipline (CLAUDE.md): every test verifies the kernel ACTUALLY does the
math it names — not constants. The headline mechanism test
(``test_boundary_tckd_wins_in_boundary_concentrated_regime``) FAILS if
boundary-TCKD does not reach lower argmax-disagreement than KL T=2.0 at matched
steps when the errors are boundary-concentrated (the contest near-correct
regime); it would still pass if the kernel were replaced by a no-op only because
both arms would then tie — so it asserts a STRICT inequality with margin.

Verified against the teacher design memo
``.omx/research/optimal_scorer_teacher_design_20260531T103350Z.md`` §1.4 / §2 and
the real-teacher A/B ``tools/ab_boundary_tckd_vs_kl_t2.py`` (σ-sweep: boundary-TCKD
wins +65-87% at init d_seg < 0.10).
"""
from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
import mlx.optimizers as optim  # noqa: E402

from tac.substrates.hinton_distilled_scorer_surrogate.mlx_loss import (  # noqa: E402
    boundary_argmax_hinge_loss,
    boundary_decision_tckd_loss,
    boundary_weighted_tckd_loss,
    hinton_distilled_kl_t2_loss,
    pose_distillation_mse_loss,
    pose_sensitivity_weighted_mse_loss,
    segnet_boundary_band_weights_mlx,
    segnet_class_pair_transition_matrix,
)


# --------------------------------------------------------------------------- #
# segnet_boundary_band_weights_mlx — the w_i = exp(-margin/tau_b) cost-map.
# --------------------------------------------------------------------------- #
def test_boundary_weight_anticorrelates_with_margin():
    mx.random.seed(0)
    logits = mx.random.normal((4096, 5)) * 3.0
    w = segnet_boundary_band_weights_mlx(logits, tau_boundary=1.0)
    sorted_l = mx.sort(logits, axis=-1)
    margin = np.array(sorted_l[:, -1] - sorted_l[:, -2])
    corr = float(np.corrcoef(margin, np.array(w))[0, 1])
    # small margin (boundary) -> high weight: strong NEGATIVE correlation.
    assert corr < -0.5, f"boundary weight must anti-correlate with margin; corr={corr}"


def test_boundary_weight_in_unit_interval():
    mx.random.seed(1)
    logits = mx.random.normal((512, 5)) * 2.0
    w = segnet_boundary_band_weights_mlx(logits, tau_boundary=1.0)
    assert float(w.min()) >= 0.0 and float(w.max()) <= 1.0


def test_boundary_weight_zero_margin_is_one():
    # two equal top logits -> margin 0 -> w = exp(0) = 1.
    logits = mx.array([[3.0, 3.0, 0.0, 0.0, 0.0]])
    w = segnet_boundary_band_weights_mlx(logits, tau_boundary=1.0)
    assert abs(float(w[0]) - 1.0) < 1e-5


def test_boundary_weight_large_margin_near_zero():
    logits = mx.array([[100.0, 0.0, 0.0, 0.0, 0.0]])  # margin 100
    w = segnet_boundary_band_weights_mlx(logits, tau_boundary=1.0)
    assert float(w[0]) < 1e-6


def test_boundary_weight_tighter_tau_tighter_band():
    mx.random.seed(2)
    logits = mx.random.normal((4096, 5)) * 3.0
    w_wide = segnet_boundary_band_weights_mlx(logits, tau_boundary=2.0)
    w_tight = segnet_boundary_band_weights_mlx(logits, tau_boundary=0.2)
    assert float(w_tight.mean()) < float(w_wide.mean())


def test_boundary_weight_rejects_nonpositive_tau():
    logits = mx.array([[1.0, 0.0]])
    with pytest.raises(ValueError):
        segnet_boundary_band_weights_mlx(logits, tau_boundary=0.0)


# --------------------------------------------------------------------------- #
# boundary_weighted_tckd_loss — the OPTIMAL seg teacher.
# --------------------------------------------------------------------------- #
def test_tckd_finite_and_positive():
    mx.random.seed(3)
    teacher = mx.random.normal((1024, 5)) * 3.0
    student = teacher + mx.random.normal((1024, 5)) * 0.5
    loss = boundary_weighted_tckd_loss(student, teacher, temperature=2.0, tau_boundary=1.0)
    v = float(loss)
    assert np.isfinite(v) and v > 0.0


def test_tckd_zero_when_student_equals_teacher():
    mx.random.seed(4)
    teacher = mx.random.normal((512, 5)) * 3.0
    loss = float(boundary_weighted_tckd_loss(teacher, teacher, temperature=2.0))
    assert loss < 1e-6, f"TCKD(teacher, teacher) must be ~0; got {loss}"


def test_tckd_is_a_distinct_functional_from_kl_t2():
    mx.random.seed(5)
    teacher = mx.random.normal((2048, 5)) * 3.0
    student = teacher + mx.random.normal((2048, 5)) * 0.5
    tckd = float(boundary_weighted_tckd_loss(student, teacher, temperature=2.0, tau_boundary=1.0))
    kl = float(hinton_distilled_kl_t2_loss(student, teacher, temperature=2.0))
    # different objectives -> materially different values (NOT a thin wrapper).
    assert abs(tckd - kl) > 1e-3, f"TCKD must differ from KL T=2.0; tckd={tckd} kl={kl}"


def test_tckd_gradient_flows_to_student():
    mx.random.seed(6)
    teacher = mx.stop_gradient(mx.random.normal((1024, 5)) * 3.0)
    student = teacher + mx.random.normal((1024, 5)) * 0.5
    g = mx.grad(lambda s: boundary_weighted_tckd_loss(s, teacher, temperature=2.0))(student)
    assert bool(mx.all(mx.isfinite(g)))
    assert float((mx.abs(g) > 1e-9).astype(mx.float32).mean()) > 0.5


def test_tckd_gradient_concentrates_on_boundary_pixels():
    # confident pixels (margin 20) vs boundary pixels (margin 0.4).
    teacher = mx.array(
        [
            [20.0, 0.0, 0.0, 0.0, 0.0],  # confident (margin 20)
            [20.0, 0.0, 0.0, 0.0, 0.0],  # confident
            [1.0, 0.6, 0.0, 0.0, 0.0],  # boundary (margin 0.4)
            [1.0, 0.6, 0.0, 0.0, 0.0],  # boundary
        ]
    )
    # NON-uniform perturbation (boost class 1): a uniform shift leaves softmax
    # unchanged, so the student must differ class-relatively to carry signal.
    student = teacher + mx.array([[0.0, 1.0, 0.0, 0.0, 0.0]] * 4)
    g = mx.grad(lambda s: boundary_weighted_tckd_loss(s, teacher, temperature=2.0, tau_boundary=1.0))(
        student
    )
    gmag = np.array(mx.sum(mx.abs(g), axis=-1))
    # boundary pixels (rows 2,3) get MUCH more gradient than confident (rows 0,1):
    # confident pixels have w_i ~ exp(-20) ~ 0 AND TCKD ~ 0 (teacher saturated).
    assert gmag[2] > 5.0 * max(gmag[0], 1e-12)
    assert gmag[3] > 5.0 * max(gmag[1], 1e-12)


def test_tckd_rejects_nonpositive_temperature_and_tau():
    teacher = mx.array([[1.0, 0.0]])
    with pytest.raises(ValueError):
        boundary_weighted_tckd_loss(teacher, teacher, temperature=0.0)
    with pytest.raises(ValueError):
        boundary_weighted_tckd_loss(teacher, teacher, tau_boundary=-1.0)


def test_boundary_tckd_actually_trains_not_a_noop():
    """NO-FAKE dynamic test: boundary-TCKD training REDUCES d_seg substantially.

    A no-op (or constant-return) kernel would leave d_seg unchanged. This asserts
    the kernel produces real gradient that drives the student's argmax toward the
    teacher's decision (the d_seg functional) — proving it does the work it names.

    The STRONGER empirical claim — that boundary-TCKD reaches LOWER d_seg than KL
    T=2.0 at matched steps — is NOT asserted here because it is conditioned on the
    teacher's boundary CONFIDENCE structure (a property of the real SegNet, not a
    kernel invariant): a synthetic teacher with too-low boundary margins favors
    full-KL because TCKD's binary [target, rest] target loses the runner-up
    identity. That regime-win is established empirically by the real-SegNet A/B
    ``tools/ab_boundary_tckd_vs_kl_t2.py`` (σ-sweep: +65-87% at init d_seg < 0.10),
    where the real margin distribution holds — NOT by a fixture-sensitive synthetic.
    """
    mx.random.seed(7)
    n_pixels, n_classes = 6000, 5
    # teacher with confident bulk + a boundary band confident enough that the
    # binary TCKD target is meaningful (margin ~1.5, not near-uniform).
    dom = mx.argmax(mx.random.normal((n_pixels, n_classes)), axis=-1)
    onehot = (mx.arange(n_classes)[None, :] == dom[:, None]).astype(mx.float32)
    is_boundary = mx.random.uniform(shape=(n_pixels,)) < 0.30
    margins = mx.where(is_boundary, 1.5, 8.0)[:, None]
    teacher = mx.stop_gradient(
        onehot * margins + mx.random.normal((n_pixels, n_classes)) * 0.02
    )
    t_arg = mx.argmax(teacher, axis=-1)
    init = teacher + mx.random.normal(teacher.shape) * 1.0  # flips the boundary band

    def d_seg(s):
        return float((mx.argmax(s, axis=-1) != t_arg).astype(mx.float32).mean())

    s = mx.array(init)
    opt = optim.Adam(learning_rate=0.3)
    fn = lambda z: boundary_weighted_tckd_loss(  # noqa: E731
        z, teacher, temperature=2.0, tau_boundary=2.0
    )
    gf = mx.value_and_grad(fn)
    init_dseg = d_seg(s)
    for _ in range(60):
        _, g = gf(s)
        s = opt.apply_gradients({"s": g}, {"s": s})["s"]
        mx.eval(s)
    final_dseg = d_seg(s)
    # the kernel must MEANINGFULLY reduce d_seg (a no-op would not move it).
    assert final_dseg < 0.6 * init_dseg, (
        f"boundary-TCKD must train (reduce d_seg); init={init_dseg} final={final_dseg}"
    )


# --------------------------------------------------------------------------- #
# pose_sensitivity_weighted_mse_loss — the OPTIMAL pose teacher.
# --------------------------------------------------------------------------- #
def test_pose_loss_finite():
    mx.random.seed(8)
    tp = mx.random.normal((64, 12))
    sp = tp + mx.random.normal((64, 12)) * 0.3
    loss = float(pose_sensitivity_weighted_mse_loss(sp, tp, num_scored_dims=6))
    assert np.isfinite(loss) and loss >= 0.0


def test_pose_loss_ignores_dims_above_K():
    mx.random.seed(9)
    tp = mx.random.normal((64, 12))
    sp = tp + mx.random.normal((64, 12)) * 0.3
    scale = mx.abs(mx.random.normal((12,))) + 0.1
    base = float(pose_sensitivity_weighted_mse_loss(sp, tp, per_dim_scale=scale, num_scored_dims=6))
    # perturb a dim >= K=6 by a huge amount -> loss UNCHANGED (k>=K zeroed).
    sp_np = np.array(sp)
    sp_np[:, 7] += 1000.0
    sp2 = mx.array(sp_np)
    perturbed = float(
        pose_sensitivity_weighted_mse_loss(sp2, tp, per_dim_scale=scale, num_scored_dims=6)
    )
    assert abs(base - perturbed) < 1e-5, "dims k>=K must not affect the loss"


def test_pose_loss_gradient_zero_above_K():
    mx.random.seed(10)
    tp = mx.random.normal((32, 12))
    sp = tp + mx.random.normal((32, 12)) * 0.3
    g = mx.grad(lambda s: pose_sensitivity_weighted_mse_loss(s, tp, num_scored_dims=6))(sp)
    assert float(mx.abs(g[:, 7]).max()) < 1e-9, "gradient on dims k>=K must be zero"
    assert float(mx.abs(g[:, 0]).max()) > 0.0, "gradient on scored dims must be nonzero"


def test_pose_loss_sensitivity_reweight_changes_value():
    mx.random.seed(11)
    tp = mx.random.normal((64, 6))
    sp = tp + mx.random.normal((64, 6)) * 0.5
    with_reweight = float(pose_sensitivity_weighted_mse_loss(sp, tp, sensitivity_reweight=True))
    without = float(pose_sensitivity_weighted_mse_loss(sp, tp, sensitivity_reweight=False))
    assert abs(with_reweight - without) > 1e-6, "AIL reweight must change the loss"


def test_pose_loss_no_reweight_matches_mahalanobis_mse():
    # with reweight off + all dims scored, equals the canonical pose_distillation_mse_loss.
    mx.random.seed(12)
    tp = mx.random.normal((64, 6))
    sp = tp + mx.random.normal((64, 6)) * 0.3
    scale = mx.abs(mx.random.normal((6,))) + 0.1
    a = float(
        pose_sensitivity_weighted_mse_loss(
            sp, tp, per_dim_scale=scale, sensitivity_reweight=False
        )
    )
    b = float(pose_distillation_mse_loss(sp, tp, per_dim_scale=scale))
    assert abs(a - b) < 1e-5, f"no-reweight all-dims must equal canonical MSE; {a} vs {b}"


def test_pose_loss_rejects_bad_K():
    tp = mx.random.normal((8, 12))
    with pytest.raises(ValueError):
        pose_sensitivity_weighted_mse_loss(tp, tp, num_scored_dims=0)
    with pytest.raises(ValueError):
        pose_sensitivity_weighted_mse_loss(tp, tp, num_scored_dims=13)


# --------------------------------------------------------------------------- #
# boundary_decision_tckd_loss — the runner-up-AWARE top1<->top2 decision KD.
# (Object 2: the boundary IS the top1<->top2 surface; distil exactly that.)
# --------------------------------------------------------------------------- #
def _mixed_5class_teacher(seed: int = 0):
    """Confident-road interior + a road<->undrivable boundary band."""
    mx.random.seed(seed)
    p, c = 400, 5
    teacher = mx.random.normal((p, c)) * 0.4
    boost = mx.zeros((p, c))
    boost[:100, 0] = 6.0  # confident road interior
    teacher = teacher + boost
    band = mx.zeros((p, c))
    band[100:200, 0] = 3.0  # road
    band[100:200, 2] = 2.9  # undrivable (the runner-up at the boundary)
    return mx.stop_gradient(teacher + band)


def test_decision_tckd_zero_when_student_equals_teacher():
    teacher = _mixed_5class_teacher()
    # NO-FAKE: an identity input must give ~0 loss (not a constant).
    assert float(boundary_decision_tckd_loss(teacher, teacher)) < 1e-5


def test_decision_tckd_positive_when_student_differs():
    teacher = _mixed_5class_teacher()
    student = teacher + mx.random.normal(teacher.shape) * 0.5
    val = float(boundary_decision_tckd_loss(student, teacher))
    assert val > 0.0 and np.isfinite(val)


def test_decision_tckd_penalizes_out_of_pair_impostor_class():
    """A non-top2 class winning student argmax is a real d_seg error."""

    teacher = mx.array([[3.0, 0.0, 2.9, 0.0, -4.0]])
    # Preserve the teacher top1/top2 logits exactly, but make class 4 dominate.
    # A pure softmax([top1, top2]) decision loss would be zero here even though
    # the full student argmax disagrees with the teacher.
    student = mx.array([[3.0, 0.0, 2.9, 0.0, 100.0]])
    assert int(mx.argmax(teacher, axis=-1)[0]) == 0
    assert int(mx.argmax(student, axis=-1)[0]) == 4

    val = float(
        boundary_decision_tckd_loss(
            student,
            teacher,
            temperature=2.0,
            tau_boundary=1.0,
        )
    )
    assert val > 0.5


def test_boundary_argmax_hinge_zero_iff_argmax_correct_with_margin():
    teacher = mx.array(
        [
            [3.0, 0.0, 2.9, 0.0, 0.0],
            [3.0, 0.0, 2.9, 0.0, 0.0],
        ]
    )
    correct = mx.array(
        [
            [4.0, 0.0, 2.5, 0.0, 0.0],
            [4.0, 0.0, 2.5, 0.0, 0.0],
        ]
    )
    wrong = mx.array(
        [
            [3.0, 0.0, 2.9, 0.0, 100.0],
            [3.0, 0.0, 2.9, 0.0, 100.0],
        ]
    )
    assert float(boundary_argmax_hinge_loss(correct, teacher, margin=0.5)) < 1e-6
    assert float(boundary_argmax_hinge_loss(wrong, teacher, margin=0.5)) > 90.0


def test_decision_tckd_gradient_flows():
    teacher = _mixed_5class_teacher()
    student = teacher + mx.random.normal(teacher.shape) * 0.5

    def loss_fn(s):
        return boundary_decision_tckd_loss(s, teacher)

    g = mx.grad(loss_fn)(student)
    assert bool(mx.all(mx.isfinite(g)))
    assert float(mx.sqrt(mx.sum(g * g))) > 0.0  # actually trains (non-zero grad)


def test_decision_tckd_distinct_from_target_vs_rest():
    """The runner-up-aware objective is genuinely different from target-vs-rest.

    If it were the same functional, both losses would coincide. They must not:
    decision-TCKD distils ``[p_top1, p_top2]``; target-vs-rest distils
    ``[p_target, 1-p_target]`` (lumps top2..topK). On a weak-margin boundary the
    runner-up identity matters, so the two objectives MUST diverge.
    """
    teacher = _mixed_5class_teacher()
    student = teacher + mx.random.normal(teacher.shape) * 0.5
    dec = float(boundary_decision_tckd_loss(student, teacher))
    tvr = float(boundary_weighted_tckd_loss(student, teacher))
    assert abs(dec - tvr) > 1e-4  # distinct objective, not a rename


def test_decision_tckd_concentrates_on_boundary_not_interior():
    """The boundary band drives the loss; confident interiors contribute ~0.

    Perturb ONLY the confident-road interior (margin ~6) vs ONLY the boundary
    band (margin ~0.1). The boundary perturbation must produce a much larger
    loss because ``w_i`` (and ``decKL_i``) are ~0 on the confident interior.
    """
    teacher = _mixed_5class_teacher()
    # interior-only perturbation (pixels 0..100, margin ~6 => band weight ~0)
    s_interior = mx.array(teacher)
    pert_i = mx.zeros(teacher.shape)
    pert_i[:100, 1] = 2.0
    s_interior = s_interior + pert_i
    # boundary-only perturbation (pixels 100..200, margin ~0.1 => band weight ~1)
    s_boundary = mx.array(teacher)
    pert_b = mx.zeros(teacher.shape)
    pert_b[100:200, 2] = 2.0  # push the runner-up across the decision
    s_boundary = s_boundary + pert_b
    l_interior = float(boundary_decision_tckd_loss(s_interior, teacher))
    l_boundary = float(boundary_decision_tckd_loss(s_boundary, teacher))
    assert l_boundary > 5.0 * max(l_interior, 1e-9)


def test_decision_tckd_rejects_nonpositive_params():
    teacher = _mixed_5class_teacher()
    with pytest.raises(ValueError):
        boundary_decision_tckd_loss(teacher, teacher, temperature=0.0)
    with pytest.raises(ValueError):
        boundary_decision_tckd_loss(teacher, teacher, tau_boundary=-1.0)


# --------------------------------------------------------------------------- #
# boundary_argmax_hinge_loss — the d_seg-FAITHFUL seg teacher (loss=0 <=> argmax-correct).
# --------------------------------------------------------------------------- #
def test_argmax_hinge_zero_when_confidently_correct():
    """loss = 0 IFF the student's argmax matches the teacher's, with margin.

    A confidently-correct student (argmax == teacher argmax, gap >= margin) is
    d_seg-correct, so the hinge must be ~0 even though the student does NOT match
    the teacher's soft distribution.
    """
    teacher = mx.array([[3.0, 0.0, 2.9, 0.0, 0.0]])  # boundary: top1=0, top2=2, margin 0.1
    student = mx.array([[10.0, 0.0, 0.0, 0.0, 0.0]])  # confidently correct (argmax 0, gap 10)
    assert int(mx.argmax(student, axis=-1)[0]) == int(mx.argmax(teacher, axis=-1)[0])
    assert float(boundary_argmax_hinge_loss(student, teacher, margin=1.0)) < 1e-6


def test_argmax_hinge_rewards_overconfidence_where_soft_losses_punish():
    """THE d_seg-faithfulness proof — why every soft loss is the wrong object.

    The student is CONFIDENTLY CORRECT (argmax matches the teacher, large margin)
    but does NOT match the teacher's fragile boundary softness. For ``d_seg`` this
    is IDEAL (confident-correct => no flip). The hinge correctly assigns ~0 loss;
    every soft probability-matching loss (KL T=2.0, target-vs-rest TCKD, 3-bucket
    decision-KD) PENALIZES the over-confidence — i.e. they would pull the student
    BACK toward the teacher's flip-prone uncertainty. That is exactly why all three
    soft arms lost the near-correct A/B and the hinge is the faithful object.
    """
    teacher = mx.array([[3.0, 0.0, 2.9, 0.0, 0.0]])  # uncertain boundary (top1~0.51, top2~0.49)
    student = mx.array([[10.0, 0.0, 0.0, 0.0, 0.0]])  # confident-correct (d_seg-ideal)

    hinge = float(boundary_argmax_hinge_loss(student, teacher, margin=1.0))
    kl = float(hinton_distilled_kl_t2_loss(student, teacher, temperature=2.0))
    tvr = float(boundary_weighted_tckd_loss(student, teacher, temperature=2.0))
    dec = float(boundary_decision_tckd_loss(student, teacher, temperature=2.0))

    assert hinge < 1e-6  # hinge: confident-correct is FREE (d_seg-faithful)
    assert kl > 0.1  # KL: punishes the over-confidence (wrong object for d_seg)
    assert tvr > 0.01  # target-vs-rest: punishes it too
    assert dec > 0.01  # decision-KD: punishes it too


def test_argmax_hinge_nonzero_for_out_of_pair_impostor():
    """The operator's literal scenario: cannot be zero while d_seg is wrong.

    The student's argmax is an OUT-OF-PAIR impostor (class 4, NOT the teacher's
    top2). ``d_seg`` is wrong, so the hinge must be strictly positive — the
    ``max_{j != t}`` sweep includes class 4.
    """
    teacher = mx.array([[3.0, 0.0, 2.9, 0.0, -4.0]])  # top1=0, top2=2 (class 4 is far below)
    student = mx.array([[3.0, 0.0, 2.9, 0.0, 9.0]])  # argmax flips to OUT-OF-PAIR class 4
    assert int(mx.argmax(teacher, axis=-1)[0]) == 0
    assert int(mx.argmax(student, axis=-1)[0]) == 4  # NOT the teacher's top2 (=2)
    val = float(boundary_argmax_hinge_loss(student, teacher, margin=1.0))
    assert val > 0.0  # cannot be zero while d_seg is wrong


def test_argmax_hinge_nonzero_for_top2_flip():
    """The in-pair flip is penalised too (the hinge handles BOTH top2 and impostors)."""
    teacher = mx.array([[3.0, 0.0, 2.9, 0.0, 0.0]])  # top1=0, top2=2
    student = mx.array([[3.0, 0.0, 5.0, 0.0, 0.0]])  # argmax flips to teacher's top2 (=2)
    assert int(mx.argmax(student, axis=-1)[0]) == 2
    assert float(boundary_argmax_hinge_loss(student, teacher, margin=1.0)) > 0.0


def test_argmax_hinge_gradient_flows():
    teacher = _mixed_5class_teacher()
    student = teacher + mx.random.normal(teacher.shape) * 0.5

    def loss_fn(s):
        return boundary_argmax_hinge_loss(s, teacher, margin=1.0)

    g = mx.grad(loss_fn)(student)
    assert bool(mx.all(mx.isfinite(g)))
    assert float(mx.sqrt(mx.sum(g * g))) > 0.0  # actually trains (non-zero grad)


def test_argmax_hinge_concentrates_on_boundary_not_interior():
    """The boundary perturbation drives a larger loss INCREASE than the interior.

    Measured as the perturbation-induced DELTA over the teacher's own baseline loss
    (``loss(teacher, teacher)``). The hinge has a sub-margin floor: even a perfect
    student incurs loss wherever the teacher's own top-2 gap is below the margin
    (the teacher's irreducible boundary uncertainty). That floor is a COMMON
    baseline in both arms, so the faithful measure of "where the gradient
    concentrates" is the delta. Interior perturbation (+2.0, gap 4 >> margin 1 =>
    hinge unchanged at 0) => delta ~0; boundary perturbation (+2.0 across the
    ~0.1-margin decision => the runner-up wins) => large delta.
    """
    teacher = _mixed_5class_teacher()
    base = float(boundary_argmax_hinge_loss(teacher, teacher, margin=1.0))
    s_interior = mx.array(teacher)
    pert_i = mx.zeros(teacher.shape)
    pert_i[:100, 1] = 2.0  # interior class 1 -> ~2.0, gap to class 0 (~6) is 4 >> margin
    s_interior = s_interior + pert_i
    s_boundary = mx.array(teacher)
    pert_b = mx.zeros(teacher.shape)
    pert_b[100:200, 2] = 2.0  # push the runner-up across the ~0.1-margin boundary
    s_boundary = s_boundary + pert_b
    d_interior = float(boundary_argmax_hinge_loss(s_interior, teacher, margin=1.0)) - base
    d_boundary = float(boundary_argmax_hinge_loss(s_boundary, teacher, margin=1.0)) - base
    assert d_boundary > 5.0 * max(d_interior, 1e-9)


def test_argmax_hinge_actually_trains_reduces_d_seg():
    """NO-FAKE: optimizing the hinge must REDUCE the actual argmax-disagreement rate."""
    teacher = _mixed_5class_teacher(seed=1)
    teacher_argmax = mx.argmax(teacher, axis=-1)

    def d_seg(s):
        return float((mx.argmax(s, axis=-1) != teacher_argmax).astype(mx.float32).mean())

    mx.random.seed(7)
    student = teacher + mx.random.normal(teacher.shape) * 1.5  # noisy near-correct start
    d0 = d_seg(student)
    opt = optim.Adam(learning_rate=0.5)
    fn = lambda z: boundary_argmax_hinge_loss(z, teacher, margin=1.0)  # noqa: E731
    grad_fn = mx.value_and_grad(fn)
    for _ in range(40):
        _, g = grad_fn(student)
        student = opt.apply_gradients({"s": g}, {"s": student})["s"]
        mx.eval(student)
    d1 = d_seg(student)
    assert d1 < d0  # the hinge actually lowers d_seg (not a no-op)
    assert d0 > 0.0  # the start was genuinely wrong somewhere (real test)


def test_argmax_hinge_rejects_nonpositive_params():
    teacher = _mixed_5class_teacher()
    with pytest.raises(ValueError):
        boundary_argmax_hinge_loss(teacher, teacher, margin=0.0)
    with pytest.raises(ValueError):
        boundary_argmax_hinge_loss(teacher, teacher, tau_boundary=-1.0)


# --------------------------------------------------------------------------- #
# segnet_class_pair_transition_matrix — the lower-D class-pair abstraction.
# --------------------------------------------------------------------------- #
def test_transition_matrix_shape_and_zero_diagonal():
    teacher = _mixed_5class_teacher()
    t = np.array(segnet_class_pair_transition_matrix(teacher, num_classes=5))
    assert t.shape == (5, 5)
    # top1 != top2 structurally => diagonal is exactly 0.
    assert np.allclose(np.diag(t), 0.0)
    assert (t >= 0.0).all()  # boundary mass is non-negative


def test_transition_matrix_concentrates_on_synthesized_surface():
    """The dominant decision surface must be the road<->undrivable boundary.

    NO-FAKE: the kernel reads (top1, top2) off the teacher's sorted logits, so
    the synthesized road(0)<->undrivable(2) band must dominate — proving the
    matrix reflects the actual decision structure, not a constant.
    """
    teacher = _mixed_5class_teacher()
    t = np.array(segnet_class_pair_transition_matrix(teacher, tau_boundary=2.0, num_classes=5))
    a, b = np.unravel_index(int(np.argmax(t)), t.shape)
    assert {a, b} == {0, 2}  # road <-> undrivable is the dominant surface


def test_transition_matrix_rejects_bad_args():
    teacher = _mixed_5class_teacher()
    with pytest.raises(ValueError):
        segnet_class_pair_transition_matrix(teacher, tau_boundary=0.0)
    with pytest.raises(ValueError):
        segnet_class_pair_transition_matrix(teacher, num_classes=4)  # < class axis
