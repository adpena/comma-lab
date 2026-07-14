# SPDX-License-Identifier: MIT
"""CPU-only contract tests for Task #494 calibrated scorer precision ladder."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from tac.local_acceleration.calibrated_fixedpoint_scorer import (
    ActivationAbsMaxCalibrator,
    FixedPointForwardPolicy,
    build_calibrated_qdq_model,
    fixedpoint_accumulator_bound,
    qmax_for_bits,
    quantize_activation_dynamic,
    quantize_weight_per_output,
)

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "fixedpoint_scorer_probe", ROOT / "tools/probe_fixedpoint_scorer_forward_n600.py"
)
assert SPEC is not None and SPEC.loader is not None
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


@pytest.mark.parametrize(
    ("bits", "qmax"),
    [
        (8, 127),
        (10, 511),
        (16, 32767),
        (24, 8_388_607),
        (25, 16_777_215),
        (26, 33_554_431),
    ],
)
def test_registered_signed_precision(bits: int, qmax: int) -> None:
    assert qmax_for_bits(bits) == qmax


def test_uniform_single_int64_ceiling_is_preregistered() -> None:
    assert PROBE.DYNAMIC_INT64_CEILING_BITS == (25, 26)
    with pytest.raises(ValueError, match="2..26"):
        qmax_for_bits(27)


def test_weight_quantization_is_per_output_channel() -> None:
    import torch

    weight = torch.tensor(
        [
            [[[1.0, -1.0], [0.5, -0.5]]],
            [[[10.0, -10.0], [5.0, -5.0]]],
        ],
        dtype=torch.float32,
    )
    dequantized, scales = quantize_weight_per_output(weight, bits=8)
    assert scales.shape == (2,)
    assert scales[1] == pytest.approx(10.0 / 127.0)
    assert scales[0] == pytest.approx(1.0 / 127.0)
    assert dequantized.shape == weight.shape


def _toy_model():
    import torch

    model = torch.nn.Sequential(
        torch.nn.Conv2d(1, 2, kernel_size=1, bias=True),
        torch.nn.ReLU(),
        torch.nn.Conv2d(2, 2, kernel_size=1, bias=False),
    ).eval()
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([[[[0.73]]], [[[-0.61]]]]))
        model[0].bias.copy_(torch.tensor([0.03, -0.02]))
        model[2].weight.copy_(
            torch.tensor([[[[0.8]], [[-0.2]]], [[[0.1]], [[0.9]]]])
        )
    return model


def test_calibration_is_frozen_and_activation_path_is_really_instrumented() -> None:
    import torch

    model = _toy_model()
    observer = ActivationAbsMaxCalibrator(model, model_kind="toy")
    with torch.inference_mode():
        model(torch.tensor([[[[-1.0, 0.5], [0.25, 1.0]]]], dtype=torch.float32))
    calibration = observer.freeze()
    digest_before = calibration.digest()
    candidate, manifest = build_calibrated_qdq_model(
        model, calibration, FixedPointForwardPolicy(bits=2)
    )
    # Values outside calibration range must be clipped by the fixed activation
    # scale.  A weight-only fake would leave this input path unchanged.
    probe = torch.tensor([[[[8.0, -8.0], [4.0, -4.0]]]], dtype=torch.float32)
    with torch.inference_mode():
        candidate_output = candidate(probe)
        weight_only_output = model(probe)
    assert not torch.equal(candidate_output, weight_only_output)
    assert calibration.digest() == digest_before
    assert manifest["quantized_operator_count"] == 2
    assert manifest["native_integer_kernel_claim"] is False


def test_dynamic_absmax_scale_is_label_free_and_does_not_clip_range() -> None:
    import torch

    value = torch.tensor([-8.0, -1.0, 0.0, 3.0, 8.0], dtype=torch.float32)
    quantized = quantize_activation_dynamic(value, bits=8)
    assert quantized[0].item() == pytest.approx(-8.0)
    assert quantized[-1].item() == pytest.approx(8.0)
    with pytest.raises(ValueError, match="non-finite"):
        quantize_activation_dynamic(torch.tensor([float("nan")]), bits=8)


def test_dynamic_qdq_manifest_names_order_invariant_scale() -> None:
    import torch

    model = _toy_model()
    observer = ActivationAbsMaxCalibrator(model, model_kind="toy")
    with torch.inference_mode():
        model(torch.ones((1, 1, 2, 2)))
    calibration = observer.freeze()
    _, manifest = build_calibrated_qdq_model(
        model,
        calibration,
        FixedPointForwardPolicy(bits=8, activation_scale_mode="dynamic_exact_absmax"),
    )
    quantized = [
        row for row in manifest["operators"] if row["precision"] != "fp32"
    ]
    assert all(row["activation_scale"] is None for row in quantized)
    assert all("commutative" in row["dynamic_scale_reduction"] for row in quantized)


def test_accumulator_manifest_distinguishes_safe_and_unsafe_widths() -> None:
    import torch

    small = torch.nn.Conv2d(3, 4, 3)
    safe = fixedpoint_accumulator_bound(small, activation_bits=8, weight_bits=8)
    assert safe["bound_kind"] == "STATIC_WORST_CASE_FAN_IN_QMAX_PRODUCT"
    assert safe["int32_safe"] is True
    huge = torch.nn.Linear(100_000, 2)
    unsafe = fixedpoint_accumulator_bound(huge, activation_bits=16, weight_bits=16)
    assert unsafe["int32_safe"] is False
    assert unsafe["minimum_signed_accumulator_bits"] > 32


def test_seg_row_positive_control_and_forced_flip() -> None:
    import torch

    reference = torch.tensor(
        [[[[2.0, 1.0]], [[1.0, 2.0]], [[0.0, 0.0]]]], dtype=torch.float32
    )
    argmax = np.asarray([[0, 1]], dtype=np.int64)
    margin = np.asarray([[1.0, 1.0]], dtype=np.float32)
    control = PROBE._seg_row(
        pair_index=0,
        reference_logits=reference,
        reference_argmax=argmax,
        baseline_margin=margin,
        candidate_logits=reference,
    )
    assert control["flips"] == 0
    assert control["uncertified_pixels"] == 0

    candidate = reference.clone()
    candidate[0, 1, 0, 0] = 3.0
    negative = PROBE._seg_row(
        pair_index=121,
        reference_logits=reference,
        reference_argmax=argmax,
        baseline_margin=margin,
        candidate_logits=candidate,
    )
    assert negative["flips"] == 1
    assert negative["uncertified_pixels"] >= 1


def _metric_row(pair_index: int, flips: int) -> dict:
    pixels = 100_000
    return {
        "pair_index": pair_index,
        "split": "calibration" if pair_index < 120 else "heldout",
        "flips": flips,
        "pixels": pixels,
        "flip_fraction": flips / pixels,
        "candidate_argmax_sha256": f"hash-{pair_index}-{flips}",
        "max_abs_logit_error": float(flips),
        "sum_squared_logit_error": float(flips),
        "logit_elements": pixels * 5,
        "uncertified_pixels": flips,
        "baseline_margin_min": 1.0,
        "baseline_margin_pair_quantiles": {
            "q0": 1.0,
            "q0.001": 1.0,
            "q0.01": 1.0,
            "q0.05": 1.0,
            "q0.5": 1.0,
        },
        "flipped_pixel_margin_quantiles": None,
    }


def test_aggregate_gates_use_heldout_and_worst_pair() -> None:
    exact = [_metric_row(index, 0) for index in range(600)]
    summary = PROBE._aggregate_rows(exact, split="heldout")
    assert summary["pairs"] == 480
    assert summary["argmax_exact_gate"] is True
    assert summary["training_tolerance_gate"] is True

    one_bad = list(exact)
    one_bad[599] = _metric_row(599, 4)  # 4e-5 worst-pair exceeds 3.3e-5.
    negative = PROBE._aggregate_rows(one_bad, split="heldout")
    assert negative["argmax_exact_gate"] is False
    assert negative["training_tolerance_gate"] is False


def test_cache_custody_requires_exact_pair_index_set_and_records_divergence() -> None:
    rows = [
        {
            "pair_index": index,
            "argmax_mismatch_pixels": int(index == 1),
            "one_thread_argmax_sha256": f"one-{index}",
            "cached_argmax_sha256": f"cache-{index}",
            "margin_max_abs_delta": float(index) * 1e-6,
        }
        for index in range(3)
    ]
    measured = PROBE._aggregate_cache_custody(rows, expected_indices={0, 1, 2})
    assert measured["status"] == "MEASURED"
    assert measured["mismatch_pairs"] == 1
    assert measured["argmax_mismatch_pixels"] == 1
    assert measured["pairs"] == measured["unique_pair_indices"] == 3
    assert (
        measured["observed_pair_indices_sha256"]
        == measured["expected_pair_indices_sha256"]
    )
    incomplete = PROBE._aggregate_cache_custody(
        [rows[0], rows[0], rows[2]], expected_indices={0, 1, 2}
    )
    assert incomplete["status"] == "INCOMPLETE"
    assert incomplete["unique_pair_indices"] == 2


def _fp32_only_admission_fixture() -> dict:
    exact = _metric_row(120, 0)
    negative = _metric_row(120, 4)
    return {
        "schema": PROBE.SCHEMA,
        "custody": {"probe_sha256": "c" * 64},
        "contract": {
            "pair_start": 120,
            "pair_count": 1,
            "include_pose": False,
            "activation_scale_mode": "fixed_calibration",
            "arms": [
                {"name": "fp32_control", "bits": 32, "mixed_head_fp32": False},
                {"name": "w8a8", "bits": 8, "mixed_head_fp32": False},
            ],
        },
        "arms": {
            "fp32_control": {"segnet_rows": [exact], "posenet_rows": []},
            "w8a8": {"segnet_rows": [negative], "posenet_rows": []},
        },
        "cache_custody": {
            "segnet_rows": [
                {
                    "pair_index": 120,
                    "argmax_mismatch_pixels": 0,
                    "one_thread_argmax_sha256": "a",
                    "cached_argmax_sha256": "a",
                    "margin_max_abs_delta": 0.0,
                }
            ]
        },
    }
def test_fp32_control_cannot_admit_the_fixedpoint_ladder() -> None:
    summary = PROBE.summarize(_fp32_only_admission_fixture())
    assert summary["arms"]["fp32_control"]["argmax_exact_admitted"] is True
    assert summary["minimum_argmax_exact_arm"] is None
    assert summary["minimum_training_tolerance_arm"] is None


def test_summary_only_finalizer_preserves_original_producer_custody(
    tmp_path: Path,
) -> None:
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(_fp32_only_admission_fixture()), encoding="utf-8")
    finalized = PROBE.finalize_existing_receipt(path)
    assert finalized["completed"] is True
    assert finalized["summary"]["minimum_argmax_exact_arm"] is None
    assert finalized["summary_finalization"]["producer_probe_sha256"] == "c" * 64
    assert finalized["summary_finalization"]["numerical_rows_recomputed"] is False


def test_arm_specs_are_preregistered_and_do_not_read_heldout_results() -> None:
    assert PROBE._arm_specs(PROBE.DEFAULT_BITS) == [
        {"name": "w8a8", "bits": 8, "mixed_head_fp32": False},
        {"name": "w10a10", "bits": 10, "mixed_head_fp32": False},
        {"name": "w12a12", "bits": 12, "mixed_head_fp32": False},
        {"name": "w14a14", "bits": 14, "mixed_head_fp32": False},
        {"name": "w16a16", "bits": 16, "mixed_head_fp32": False},
        {"name": "w18a18", "bits": 18, "mixed_head_fp32": False},
        {"name": "w20a20", "bits": 20, "mixed_head_fp32": False},
        {"name": "w22a22", "bits": 22, "mixed_head_fp32": False},
        {"name": "w24a24", "bits": 24, "mixed_head_fp32": False},
        {"name": "w8a8_head_fp32", "bits": 8, "mixed_head_fp32": True},
    ]
