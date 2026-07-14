from __future__ import annotations

import numpy as np
import pytest

from tac.scorer_surrogate import whole_teacher_distilled_student as student


def test_numpy_analytic_input_vjp_matches_central_finite_difference() -> None:
    architecture = student.architecture_for_size("tiny")
    parameters = student.initialize_student_parameters(architecture, seed=455)
    rng = np.random.default_rng(17)
    frame = rng.uniform(32.0, 224.0, size=(1, 3, 3, 4)).astype(np.float32)
    cotangent = rng.standard_normal((1, 4, 3, 4), dtype=np.float32)
    analytic = student.student_input_vjp_numpy(frame, architecture, parameters, cotangent)

    epsilon = np.float32(0.05)
    finite_difference = np.zeros_like(frame)
    for index in np.ndindex(frame.shape):
        positive = frame.copy()
        negative = frame.copy()
        positive[index] += epsilon
        negative[index] -= epsilon
        positive_value = np.sum(
            student.student_forward_numpy(positive, architecture, parameters) * cotangent,
            dtype=np.float64,
        )
        negative_value = np.sum(
            student.student_forward_numpy(negative, architecture, parameters) * cotangent,
            dtype=np.float64,
        )
        finite_difference[index] = (positive_value - negative_value) / (2.0 * float(epsilon))

    np.testing.assert_allclose(analytic, finite_difference, rtol=8.0e-3, atol=2.0e-5)
    cosine = float(
        np.dot(analytic.reshape(-1), finite_difference.reshape(-1))
        / (np.linalg.norm(analytic) * np.linalg.norm(finite_difference))
    )
    assert cosine > 0.9999


def test_torch_debug_forward_and_vjp_match_numpy_reference() -> None:
    torch = pytest.importorskip("torch")
    architecture = student.architecture_for_size("tiny")
    parameters = student.initialize_student_parameters(architecture, seed=73)
    rng = np.random.default_rng(23)
    frame = rng.uniform(0.0, 255.0, size=(1, 3, 4, 5)).astype(np.float32)
    cotangent = rng.standard_normal((1, 4, 4, 5), dtype=np.float32)

    numpy_output = student.student_forward_numpy(frame, architecture, parameters)
    torch_output = student.student_forward_torch(frame, architecture, parameters).detach().cpu().numpy()
    np.testing.assert_allclose(torch_output, numpy_output, rtol=3.0e-6, atol=3.0e-6)

    numpy_vjp = student.student_input_vjp_numpy(frame, architecture, parameters, cotangent)
    torch_vjp = (
        student.student_input_vjp_torch(frame, architecture, parameters, cotangent)
        .detach()
        .cpu()
        .numpy()
    )
    np.testing.assert_allclose(torch_vjp, numpy_vjp, rtol=5.0e-6, atol=5.0e-7)
    assert torch_vjp.dtype == np.float32
    assert torch is not None


def test_quotient_ce_input_vjp_matches_torch_autograd() -> None:
    torch = pytest.importorskip("torch")
    functional = pytest.importorskip("torch.nn.functional")
    architecture = student.architecture_for_size("tiny")
    parameters = student.initialize_student_parameters(architecture, seed=91)
    rng = np.random.default_rng(31)
    frame = rng.uniform(0.0, 255.0, size=(1, 3, 3, 5)).astype(np.float32)
    labels = rng.integers(0, 5, size=(1, 3, 5), dtype=np.int64)

    numpy_costate = student.student_ce_input_vjp_numpy(frame, architecture, parameters, labels)
    frame_t = torch.tensor(frame, requires_grad=True)
    quotient_t = student.student_forward_torch(frame_t, architecture, parameters)
    logits_t = student.logits5_from_quotient4_torch(quotient_t)
    loss = functional.cross_entropy(logits_t, torch.tensor(labels))
    (torch_costate,) = torch.autograd.grad(loss, frame_t)
    np.testing.assert_allclose(
        numpy_costate,
        torch_costate.detach().numpy(),
        rtol=8.0e-6,
        atol=8.0e-8,
    )


def test_vjp_metrics_use_full_vector_and_expose_worst_pair() -> None:
    rng = np.random.default_rng(101)
    teacher = rng.standard_normal((1, 3, 4, 5), dtype=np.float32)
    mask = np.zeros((4, 5), dtype=np.bool_)
    mask[:, :2] = True
    good = student.vjp_pair_metrics("good", teacher, teacher, boundary_mask=mask)
    bad = student.vjp_pair_metrics("bad", teacher, -teacher, boundary_mask=mask)
    assert good["cosine"] == pytest.approx(1.0)
    assert good["relative_l2"] == pytest.approx(0.0)
    assert bad["cosine"] == pytest.approx(-1.0)
    assert bad["relative_l2"] == pytest.approx(2.0)
    assert good["boundary_diagnostic"]["compared_elements"] == 1 * 3 * 4 * 2

    summary = student.aggregate_vjp_pair_metrics([good, bad])
    assert summary["worst_cosine_assignment_id"] == "bad"
    assert summary["worst_relative_l2_assignment_id"] == "bad"
    assert summary["worst_cosine"] == pytest.approx(-1.0)
    assert summary["worst_relative_l2"] == pytest.approx(2.0)


def test_zero_vector_fidelity_fails_closed_as_cosine_is_undefined() -> None:
    zeros = np.zeros((1, 3, 2, 2), dtype=np.float32)
    nonzero = np.ones_like(zeros)

    for assignment_id, teacher, candidate in (
        ("both-zero", zeros, zeros),
        ("zero-student", nonzero, zeros),
        ("zero-teacher", zeros, nonzero),
    ):
        with pytest.raises(student.StudentContractError, match="undefined"):
            student.vjp_pair_metrics(assignment_id, teacher, candidate)
