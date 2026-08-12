from __future__ import annotations

import numpy as np
import pytest
import torch

from experiments import ddm_se1_shipping_axis_survival_resolve as se1


def test_packet_roundtrip_and_repeat_are_exact() -> None:
    module = b"parse-backed-module-payload"
    scales = np.array([1.0, 0.5, 0.25, 0.125, 0.0625], dtype=np.float32)
    first = se1.build_packet(module, scales)
    parsed = se1.parse_packet(first)
    second = se1.build_packet(parsed.module, parsed.scales)
    assert first == second
    assert parsed.module == module
    assert np.array_equal(parsed.scales, scales)


def test_packet_rejects_invalid_scales() -> None:
    with pytest.raises(ValueError):
        se1.build_packet(b"module", np.ones(4, dtype=np.float32))
    with pytest.raises(ValueError):
        se1.build_packet(b"module", np.array([1, 1, 1, 1, -1], dtype=np.float32))


def test_score_delta_recomputes_all_three_terms() -> None:
    row = se1.score_delta(0.001, 0.0009, 0.0002, 0.00019, 100)
    assert row["seg"] == pytest.approx(-0.01)
    assert row["pose"] < 0.0
    assert row["rate"] == pytest.approx(2500 / se1.RATE_DENOMINATOR)
    assert row["total"] == pytest.approx(row["seg"] + row["pose"] + row["rate"])


def test_admission_requires_joint_pose_and_robust_margin() -> None:
    row = {
        "realized_joint_delta_section_additive": -0.01,
        "pose_delta_stratified_n32": 1e-6,
        "robust_beneficial_flips": 1,
    }
    assert se1.admission(row)
    assert not se1.admission({**row, "pose_delta_stratified_n32": se1.POSE_GATE})
    assert not se1.admission({**row, "realized_joint_delta_section_additive": 0.0})
    assert not se1.admission({**row, "robust_beneficial_flips": 0})


def test_c1_teacher_changes_weighting_not_evaluator_labels() -> None:
    logits = torch.zeros((1, 5, 1, 2), requires_grad=True)
    labels = torch.tensor([[[0, 1]]])
    baseline = torch.tensor([[[2, 2]]])
    events = torch.tensor([[[True, False]]])
    loss, terms = se1.c1_delta_hinge_objective(torch, logits, labels, baseline, events)
    loss.backward()
    assert terms["event_wrong_pixels"] == 1
    assert float(logits.grad[0, 0, 0, 0]) < 0.0
    assert float(logits.grad[0, 1, 0, 1]) < 0.0


def test_class_survival_reports_delta_margin_mass() -> None:
    logits = np.zeros((1, 5, 1, 2), dtype=np.float32)
    baseline = np.array([[[1, 1]]], dtype=np.uint8)
    gt = np.array([[[0, 0]]], dtype=np.uint8)
    c1_events = np.array([[[True, False]]])
    logits[:, 0] = 1.0
    logits[0, 1, 0, 0] = 1.0 - 2.0 * se1.DELTA
    logits[0, 1, 0, 1] = 1.0 - 0.5 * se1.DELTA
    metrics, prediction = se1.class_survival_metrics(logits, baseline, gt, c1_events, np.array([600], dtype=np.int64))
    assert np.array_equal(prediction, gt)
    assert metrics["beneficial_flips"] == 2
    assert metrics["robust_beneficial_flips"] == 1
    assert metrics["delta_margin_mass"] == pytest.approx(0.5)
    assert metrics["c1_event_robust_beneficial_flips"] == 1
    assert metrics["per_class"][0]["robust_beneficial_flips"] == 1
