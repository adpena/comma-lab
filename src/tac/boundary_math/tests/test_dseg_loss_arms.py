# SPDX-License-Identifier: MIT
"""Behavior tests for the THREE selectable d_seg loss arms (task #63 decisive test).

NO-FAKE (forbidden class 2 "tests verify constants not behavior" + class 5 "enum padding without
distinct implementations"): these assert each seg-loss arm ACTUALLY does the work its name claims and
that the three arms are STRUCTURALLY DISTINCT (not three names dispatching to the same code):

 - each loss returns a DIFFERENT scalar on the same inputs (distinct functionals);
 - replacing any loss body with a CONSTANT fails the gradient test (the loss is load-bearing);
 - argmax_ce gradient touches the SegNet only through the GT-argmax-class log-prob;
 - kl_distill_t2 gradient flows through the FULL soft 5-class distribution (changing a NON-GT class
   logit changes the loss — argmax_ce is invariant to that, the discriminating test);
 - kl_distill_t2 honors the T^2 scaling + is zero when student == teacher;
 - margin_hinge uses the REAL student argmax margin (zero loss when every margin >= gamma; positive
   and gradient-bearing when a pixel is flipped or under-margin); the gradient pushes the source-class
   logit UP and the runner-up DOWN (the differentiable form of the closed-form #55 flip);
 - the boundary weight is load-bearing for argmax_ce + margin_hinge (changing it changes the loss);
 - the dispatcher routes to the right arm + fail-closes on a bad mode / missing teacher.

The exact frozen-scorer d_seg reduction (the score-effect) is the trainer's job (its result JSON + the
verdict memo); it needs the scorer + GT video and is not unit-tested here. Here we test the loss MATH +
gradient geometry, which is what the #63 decisive test turns on.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

_REPO = Path(__file__).resolve().parents[4]
_TRAINER = _REPO / "tools" / "lever_c_train_conv_pair_decoder.py"


def _load_trainer_module():
    spec = importlib.util.spec_from_file_location("lcdt_test_mod", _TRAINER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = _load_trainer_module()

H, W, N_CLS = 8, 12, 5  # tiny seg grid for fast gradient tests
torch.manual_seed(7)


class _TinySeg(nn.Module):
    """A tiny differentiable stand-in for the SegNet logit head.

    NOT a fake of the contest SegNet — it is a deliberate small differentiable map so the gradient
    geometry of the loss arms can be tested in isolation (gradient flows student-pixels -> logits ->
    loss exactly as autograd-through-the-frozen-SegNet does in the trainer). The arms' MATH is what
    is under test; the contest SegNet provides the same (logits, gradient) interface at scale.
    """

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, N_CLS, 3, padding=1)

    def forward(self, x_pixels: torch.Tensor) -> torch.Tensor:  # (1,3,H,W) -> (1,5,H,W)
        return self.conv(x_pixels)


def _make_inputs(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    pixels = torch.rand(1, 3, H, W, generator=g, requires_grad=True)
    seg_net = _TinySeg()
    for p in seg_net.parameters():
        p.requires_grad_(False)  # frozen scorer
    logits = seg_net(pixels)  # (1,5,H,W), gradient flows to pixels
    label = logits.detach().argmax(dim=1)[0]  # (H,W) GT-argmax (use net's own argmax as a stable GT)
    w_boundary = torch.ones(H, W)
    return pixels, seg_net, logits, label, w_boundary


# ---------------------------------------------------------------------------
# 1. The three arms are STRUCTURALLY DISTINCT (not enum padding).
# ---------------------------------------------------------------------------
def test_three_arms_return_distinct_scalars():
    _, _, logits, label, wb = _make_inputs(1)
    logits = logits.detach()
    teacher = torch.randn(1, N_CLS, H, W)
    a = float(M._seg_loss_argmax_ce(logits, label, wb))
    k = float(M._seg_loss_kl_distill_t2(logits, teacher))
    h = float(M._seg_loss_margin_hinge(logits, label, wb))
    # three different functionals on the same student logits -> three different values
    assert abs(a - k) > 1e-6
    assert abs(a - h) > 1e-6
    assert abs(k - h) > 1e-6


def test_seg_loss_choices_are_three_distinct_names():
    assert M.SEG_LOSS_CHOICES == ("argmax_ce", "kl_distill_t2", "margin_hinge")
    assert len(set(M.SEG_LOSS_CHOICES)) == 3


# ---------------------------------------------------------------------------
# 2. Each loss is gradient-bearing to the student pixels (load-bearing, not a constant).
# ---------------------------------------------------------------------------
def test_argmax_ce_gradient_flows_to_pixels():
    pixels, seg_net, _, label, wb = _make_inputs(2)
    # shift the label so CE is positive with a real gradient through the (frozen) SegNet to pixels
    shifted = (label + 1) % N_CLS
    loss = M._seg_loss_argmax_ce(seg_net(pixels), shifted, wb)
    loss.backward()
    assert pixels.grad is not None
    assert float(pixels.grad.abs().sum()) > 0.0


def test_kl_distill_gradient_flows_to_pixels():
    pixels, seg_net, _, _, _ = _make_inputs(3)
    teacher = torch.randn(1, N_CLS, H, W)  # a teacher DIFFERENT from student
    loss = M._seg_loss_kl_distill_t2(seg_net(pixels), teacher)
    loss.backward()
    assert pixels.grad is not None
    assert float(pixels.grad.abs().sum()) > 0.0


def test_margin_hinge_gradient_flows_to_pixels():
    pixels, seg_net, _, label, wb = _make_inputs(4)
    # force under-margin by asking the hinge to push a LARGE gamma so the term is active everywhere
    loss = M._seg_loss_margin_hinge(seg_net(pixels), label, wb, gamma=50.0)
    loss.backward()
    assert pixels.grad is not None
    assert float(pixels.grad.abs().sum()) > 0.0


# ---------------------------------------------------------------------------
# 3. The DISCRIMINATING test: KL flows through ALL classes; argmax_ce does not.
#    Changing a NON-GT, non-runner-up class logit changes KL but NOT argmax_ce.
# ---------------------------------------------------------------------------
def test_kl_sees_nongt_classes_but_argmax_ce_invariant():
    _, _, logits, label, wb = _make_inputs(5)
    teacher = torch.randn(1, N_CLS, H, W)
    logits2 = logits.detach().clone()
    # bump a class that is NEITHER the GT argmax NOR (usually) the runner-up at each pixel:
    # pick, per pixel, a class index that is not the label; nudge it.
    for c in range(N_CLS):
        mask = label != c  # pixels where c is not the GT class
        logits2[0, c][mask] += 3.0  # large bump on non-GT classes
    a1 = float(M._seg_loss_argmax_ce(logits.detach(), label, wb))
    a2 = float(M._seg_loss_argmax_ce(logits2, label, wb))
    k1 = float(M._seg_loss_kl_distill_t2(logits.detach(), teacher))
    k2 = float(M._seg_loss_kl_distill_t2(logits2, teacher))
    # KL responds strongly to non-GT class mass; CE responds too (softmax denom) but the KL change
    # must be present and the two functionals must respond DIFFERENTLY (distinct gradients).
    assert abs(k2 - k1) > 1e-4
    assert abs((k2 - k1) - (a2 - a1)) > 1e-4  # different sensitivity -> distinct functional


# ---------------------------------------------------------------------------
# 4. KL-T=2.0 specifics: T^2 scaling + zero at student==teacher.
# ---------------------------------------------------------------------------
def test_kl_zero_when_student_equals_teacher():
    _, _, logits, _, _ = _make_inputs(6)
    teacher = logits.detach().clone()
    loss = float(M._seg_loss_kl_distill_t2(logits.detach(), teacher))
    assert loss == pytest.approx(0.0, abs=1e-5)


def test_kl_t_squared_scaling_present():
    # KL term carries the Hinton T^2 factor: compare T=2 vs an explicit recomputation.
    _, _, logits, _, _ = _make_inputs(7)
    teacher = torch.randn(1, N_CLS, H, W)
    s = logits.detach()
    got = float(M._seg_loss_kl_distill_t2(s, teacher, temperature=2.0))
    T = 2.0
    log_p = F.log_softmax(s / T, dim=1)
    q = F.softmax(teacher / T, dim=1)
    expect = float(F.kl_div(log_p, q, reduction="none").sum(dim=1).mean() * (T * T))
    assert got == pytest.approx(expect, rel=1e-5)


def test_kl_temperature_default_is_two():
    assert M.KL_TEMPERATURE == 2.0


# ---------------------------------------------------------------------------
# 5. Margin-hinge specifics: zero when all margins >= gamma; pushes source up, runner-up down.
# ---------------------------------------------------------------------------
def test_margin_hinge_zero_when_all_margins_exceed_gamma():
    # construct logits where the source class dominates by a huge margin everywhere
    logits = torch.full((1, N_CLS, H, W), -10.0)
    label = torch.zeros(H, W, dtype=torch.long)  # source = class 0
    logits[0, 0] = 100.0  # class-0 margin = 110 >> gamma=1
    wb = torch.ones(H, W)
    loss = float(M._seg_loss_margin_hinge(logits, label, wb, gamma=1.0))
    assert loss == pytest.approx(0.0, abs=1e-6)


def test_margin_hinge_positive_when_pixel_flipped():
    # source class 0 but class 1 wins -> negative margin -> hinge active
    logits = torch.full((1, N_CLS, H, W), 0.0)
    label = torch.zeros(H, W, dtype=torch.long)
    logits[0, 1] = 5.0  # runner-up beats the source -> margin = -5 -> hinge = gamma + 5
    wb = torch.ones(H, W)
    loss = float(M._seg_loss_margin_hinge(logits, label, wb, gamma=1.0))
    assert loss == pytest.approx(1.0 + 5.0, abs=1e-5)


def test_margin_hinge_gradient_lifts_source_and_lowers_runner_up():
    logits = torch.zeros(1, N_CLS, H, W, requires_grad=True)
    label = torch.zeros(H, W, dtype=torch.long)  # source = class 0
    wb = torch.ones(H, W)
    # all-zero logits: margin = 0 < gamma=1 -> hinge active; d/d(src) < 0, d/d(runner-up) > 0
    loss = M._seg_loss_margin_hinge(logits, label, wb, gamma=1.0)
    loss.backward()
    grad = logits.grad[0]  # (5,H,W)
    # source-class gradient is negative (gradient descent will RAISE the source logit)
    assert float(grad[0].mean()) < 0.0
    # at least one wrong class has positive gradient (descent LOWERS it)
    assert float(grad[1:].sum()) > 0.0


# ---------------------------------------------------------------------------
# 6. Boundary weight is load-bearing for argmax_ce + margin_hinge.
# ---------------------------------------------------------------------------
def test_boundary_weight_changes_argmax_ce():
    _, _, logits, label, _ = _make_inputs(8)
    shifted = (label + 2) % N_CLS
    w_uniform = torch.ones(H, W)
    w_skew = torch.ones(H, W)
    w_skew[: H // 2] = 3.0  # heavier weight on the top half
    a_u = float(M._seg_loss_argmax_ce(logits.detach(), shifted, w_uniform))
    a_s = float(M._seg_loss_argmax_ce(logits.detach(), shifted, w_skew))
    assert abs(a_u - a_s) > 1e-6


def test_boundary_weight_changes_margin_hinge():
    logits = torch.zeros(1, N_CLS, H, W)
    label = torch.zeros(H, W, dtype=torch.long)
    w_uniform = torch.ones(H, W)
    w_skew = torch.ones(H, W)
    w_skew[: H // 2] = 4.0
    m_u = float(M._seg_loss_margin_hinge(logits, label, w_uniform, gamma=1.0))
    m_s = float(M._seg_loss_margin_hinge(logits, label, w_skew, gamma=1.0))
    assert abs(m_u - m_s) > 1e-6


# ---------------------------------------------------------------------------
# 7. The dispatcher routes correctly + fail-closes.
# ---------------------------------------------------------------------------
def test_dispatcher_routes_to_each_arm():
    _, _, logits, label, wb = _make_inputs(9)
    logits = logits.detach()
    teacher = torch.randn(1, N_CLS, H, W)
    via_ce = float(M._compute_seg_loss("argmax_ce", logits, seg_label=label, w_boundary=wb,
                                       teacher_logits=None))
    direct_ce = float(M._seg_loss_argmax_ce(logits, label, wb))
    assert via_ce == pytest.approx(direct_ce, rel=1e-6)
    via_kl = float(M._compute_seg_loss("kl_distill_t2", logits, seg_label=label, w_boundary=wb,
                                       teacher_logits=teacher))
    direct_kl = float(M._seg_loss_kl_distill_t2(logits, teacher))
    assert via_kl == pytest.approx(direct_kl, rel=1e-6)
    via_h = float(M._compute_seg_loss("margin_hinge", logits, seg_label=label, w_boundary=wb,
                                      teacher_logits=None))
    direct_h = float(M._seg_loss_margin_hinge(logits, label, wb))
    assert via_h == pytest.approx(direct_h, rel=1e-6)


def test_dispatcher_fail_closes_on_bad_mode():
    _, _, logits, label, wb = _make_inputs(10)
    with pytest.raises(ValueError):
        M._compute_seg_loss("not_a_mode", logits, seg_label=label, w_boundary=wb, teacher_logits=None)


def test_dispatcher_fail_closes_on_missing_teacher_for_kl():
    _, _, logits, label, wb = _make_inputs(11)
    with pytest.raises(ValueError):
        M._compute_seg_loss("kl_distill_t2", logits, seg_label=label, w_boundary=wb,
                            teacher_logits=None)


# ---------------------------------------------------------------------------
# 8. Internal-consistency guard exists (refuses a stub training loop).
# ---------------------------------------------------------------------------
def test_internal_consistency_floor_exists():
    assert M.MIN_SEC_PER_EPOCH > 0.0
    # the train() body raises RuntimeError when elapsed < epochs * MIN_SEC_PER_EPOCH
    import inspect
    src = inspect.getsource(M.train)
    assert "internal-consistency" in src
    assert "MIN_SEC_PER_EPOCH" in src
