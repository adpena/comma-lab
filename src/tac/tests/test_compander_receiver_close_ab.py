"""Synthetic receipt tests for the future compander receiver-close A/B stub."""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
TOOL = REPO / "tools/probe_compander_receiver_close_ab.py"
SPEC = importlib.util.spec_from_file_location("_compander_receiver_close_ab", TOOL)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def _arm(label: str, *, treatment: bool = False) -> dict:
    archive_sha = ("b" if treatment else "a") * 64
    per_class = {
        "Road": 0.010,
        "Lane": 0.020 - (0.002 if treatment else 0.0),
        "Undrivable": 0.030,
        "Movable": 0.040,
        "MyCar": 0.050,
    }
    arm = {
        "label": label,
        "n_pairs": 600,
        "optimizer_steps": 12345,
        "seed": 7,
        "axis": "[contest-CPU]",
        "config_custody": {"sha256": "c" * 64},
        "archive_custody": {"bytes": 123456, "sha256": archive_sha},
        "receiver_custody": {
            "parseback_passed": True,
            "archive_sha256": archive_sha,
            "decoded_sha256": "d" * 64,
            "prearchive_reference_sha256": "d" * 64,
            "runtime_sha256": "e" * 64,
        },
        "per_class_d_seg": per_class,
        "d_pose": 0.010 - (0.001 if treatment else 0.0),
    }
    if treatment:
        arm["dsl_lever_factories"] = ["MarginCompandedGroundChart"]
        arm["chart_payload_custody"] = {
            "counted_in_total_archive_bytes": True,
            "bytes": 32,
            "sha256": "f" * 64,
            "containing_archive_sha256": archive_sha,
        }
    return arm


def test_matched_receiver_close_receipts_report_lane_primary_effect() -> None:
    result = probe.compare_receipts(_arm("control"), _arm("treatment", treatment=True))
    assert result["matched"] == {
        "n_pairs": 600,
        "optimizer_steps": 12345,
        "seed": 7,
        "total_archive_bytes": 123456,
    }
    assert result["primary_lane_d_seg_delta_treatment_minus_control"] == pytest.approx(-0.002)
    assert result["d_pose_delta_treatment_minus_control"] == pytest.approx(-0.001)
    assert result["pose_nonworsening"] is True
    assert result["rate_matched_exactly"] is True
    assert result["score_claim"] is False
    assert result["promotion_claim"] is False


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda arm: arm.update(optimizer_steps=999), "optimizer steps"),
        (lambda arm: arm.update(seed=9), "seeds"),
        (lambda arm: arm["archive_custody"].update(bytes=999), "archive bytes"),
        (
            lambda arm: arm["receiver_custody"].update(decoded_sha256="9" * 64),
            "receiver decode differs",
        ),
        (
            lambda arm: arm["chart_payload_custody"].update(
                counted_in_total_archive_bytes=False
            ),
            "not explicitly receiver-counted",
        ),
        (lambda arm: arm.update(dsl_lever_factories=[]), "MarginCompandedGroundChart"),
        (lambda arm: arm["per_class_d_seg"].pop("Lane"), "per_class_d_seg.Lane"),
    ],
)
def test_refuses_unmatched_or_uncounted_treatment(mutation, message: str) -> None:
    control = _arm("control")
    treatment = copy.deepcopy(_arm("treatment", treatment=True))
    mutation(treatment)
    with pytest.raises(probe.ReceiptRefusal, match=message):
        probe.compare_receipts(control, treatment)
