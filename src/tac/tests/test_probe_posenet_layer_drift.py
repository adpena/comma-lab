from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


def _load_tool(name: str):
    repo_root = Path(__file__).resolve().parents[3]
    tool_path = repo_root / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, tool_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {tool_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_compare_tensors_reports_rms_and_relative_drift() -> None:
    mod = _load_tool("probe_posenet_layer_drift")
    a = torch.tensor([1.0, 2.0, 4.0])
    b = torch.tensor([1.0, 1.5, 5.0])

    comparison = mod.compare_tensors(a, b)

    assert comparison["shape_match"] is True
    assert comparison["numel"] == 3
    assert comparison["max_abs"] == 1.0
    assert 0.645 < comparison["rms_abs"] < 0.646
    assert comparison["max_rel"] == 0.25


def test_capture_tensor_clones_before_inplace_mutation() -> None:
    mod = _load_tool("probe_posenet_layer_drift")
    live = torch.tensor([-2.0, 3.0])

    captured = mod.capture_tensor(live)
    live.relu_()

    assert captured.tolist() == [-2.0, 3.0]
    assert live.tolist() == [0.0, 3.0]


def test_flatten_tensors_handles_pose_output_dict() -> None:
    mod = _load_tool("probe_posenet_layer_drift")
    output = {
        "pose": torch.zeros(1, 12),
        "ignored": {"nested": "not a tensor"},
    }

    rows = mod.flatten_tensors(output, prefix="pose_output")

    assert [(key, tuple(tensor.shape)) for key, tensor in rows] == [
        ("pose_output.pose", (1, 12))
    ]


def test_select_named_modules_can_target_semantic_boundaries() -> None:
    mod = _load_tool("probe_posenet_layer_drift")
    model = torch.nn.Sequential(
        torch.nn.Conv2d(1, 2, 1),
        torch.nn.Sequential(torch.nn.ReLU(), torch.nn.Conv2d(2, 2, 1)),
    )

    selected = mod.select_named_modules(
        model,
        module_regex=r"^(0|1)$",
        leaf_only=False,
        max_modules=None,
    )

    assert [name for name, _module in selected] == ["0", "1"]


def test_evaluator_rgb_to_channel_first_converts_loader_contract() -> None:
    mod = _load_tool("probe_posenet_layer_drift")
    x = torch.arange(1 * 2 * 4 * 5 * 3).reshape(1, 2, 4, 5, 3)

    out = mod.evaluator_rgb_to_channel_first(x)

    assert tuple(out.shape) == (1, 2, 3, 4, 5)
    assert out[0, 1, 2, 3, 4].item() == x[0, 1, 3, 4, 2].item()
    assert out.is_contiguous()


def test_evaluator_rgb_to_channel_first_rejects_wrong_contract() -> None:
    mod = _load_tool("probe_posenet_layer_drift")
    with torch.no_grad():
        bad = torch.zeros(1, 2, 3, 4, 5)

    import pytest

    with pytest.raises(ValueError, match="evaluator_rgb"):
        mod.evaluator_rgb_to_channel_first(bad)


def test_module_inventory_records_attention_like_modules() -> None:
    mod = _load_tool("probe_posenet_layer_drift")

    class Tiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = torch.nn.Conv2d(1, 1, 1)
            self.softmax = torch.nn.Softmax(dim=-1)

    inventory = mod.module_inventory(Tiny())

    assert inventory["module_type_counts"]["Conv2d"] == 1
    assert inventory["attention_or_softmax_named_modules"] == [
        {"name": "softmax", "type": "Softmax"}
    ]


def test_probe_report_is_non_promotable_when_cuda_unavailable() -> None:
    mod = _load_tool("probe_posenet_layer_drift")
    args = mod.parse_args(
        [
            "--device-a",
            "cpu",
            "--device-b",
            "meta",
            "--max-modules",
            "1",
        ]
    )

    report = mod.build_probe_report(args)

    assert report["score_claim"] is False
    assert report["score_claim_valid"] is False
    assert report["promotion_eligible"] is False
    assert report["rank_or_kill_eligible"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    assert report["evidence_grade"] == "diagnostic"
    assert report["diagnostic_kind"] == "posenet_layer_drift_probe"
    assert report["comparison_available"] is False
    assert report["device_pair_custody"]["score_claim_axis"] == "none"
    assert report["device_pair_custody"]["contest_cuda_claim"] is False
    assert report["device_pair_custody"]["contest_cpu_claim"] is False
    assert "selected_modules" in report
    assert "first exceeding activation" in " ".join(report["interpretation_guardrails"])


def test_invalid_device_string_fails_closed() -> None:
    mod = _load_tool("probe_posenet_layer_drift")

    available, reason = mod.device_available("definitely_not_a_device")

    assert available is False
    assert "invalid torch device" in reason


def test_cuda_index_outside_device_count_fails_closed(monkeypatch) -> None:
    mod = _load_tool("probe_posenet_layer_drift")
    monkeypatch.setattr(mod.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(mod.torch.cuda, "device_count", lambda: 1)

    available, reason = mod.device_available("cuda:7")

    assert available is False
    assert "outside available device_count=1" in reason


def test_missing_input_tensor_is_non_promotable(monkeypatch, tmp_path) -> None:
    mod = _load_tool("probe_posenet_layer_drift")
    missing = tmp_path / "missing.pt"
    args = mod.parse_args(["--input-tensor", str(missing), "--device-b", "meta"])
    monkeypatch.setattr(
        mod,
        "detect_artifacts",
        lambda _args: {
            "upstream_modules_py": True,
            "posenet_weights_exist": True,
            "timm_available": True,
            "safetensors_available": True,
            "segmentation_models_pytorch_available": True,
            "input_tensor_exists": False,
        },
    )

    report = mod.build_probe_report(args)

    assert report["comparison_available"] is False
    assert report["score_claim_valid"] is False
    assert report["promotion_eligible"] is False
    assert "input_tensor_exists=false" in report["comparison_unavailable_reasons"]


def test_unavailable_comparison_inventory_error_still_emits_json(monkeypatch) -> None:
    mod = _load_tool("probe_posenet_layer_drift")
    args = mod.parse_args(["--device-a", "cpu", "--device-b", "meta"])

    def _raise_inventory(_device):
        raise RuntimeError("inventory fixture failure")

    monkeypatch.setattr(mod, "_load_upstream_posenet", _raise_inventory)

    report = mod.build_probe_report(args)

    assert report["comparison_available"] is False
    assert report["score_claim_valid"] is False
    assert report["promotion_eligible"] is False
    assert "module_inventory_error: RuntimeError" in report["module_inventory_unavailable_reason"]
