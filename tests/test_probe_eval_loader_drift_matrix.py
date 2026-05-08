from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_tool():
    repo_root = Path(__file__).resolve().parents[1]
    tool_path = repo_root / "tools" / "probe_eval_loader_drift.py"
    spec = importlib.util.spec_from_file_location("probe_eval_loader_drift_matrix", tool_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {tool_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _artifact_status(**overrides):
    status = {
        "upstream_frame_utils_py": True,
        "upstream_modules_py": True,
        "posenet_weights_exist": True,
        "segnet_weights_exist": True,
        "video_names_file_exists": True,
        "data_dir_exists": True,
        "sampled_video_names": ["0.mkv"],
        "sampled_video_paths": ["data/0.mkv"],
        "sample_videos_exist": True,
        "sample_video_error": None,
        "pyav_available": True,
        "cuda_available": False,
        "dali_available": False,
        "cuda_dali_runtime_available": None,
        "cuda_dali_runtime_unavailable_reason": None,
        "timm_available": True,
        "einops_available": True,
        "safetensors_torch_available": True,
        "segmentation_models_pytorch_available": True,
    }
    status.update(overrides)
    return status


def test_intended_cells_record_four_labels_and_fail_closed_axes(monkeypatch) -> None:
    mod = _load_tool()
    monkeypatch.setattr(mod, "detect_artifacts", lambda _args: _artifact_status())
    called = False

    def _unexpected_cuda_probe():
        nonlocal called
        called = True
        return True, None

    monkeypatch.setattr(mod, "_cuda_dali_available", _unexpected_cuda_probe)
    args = mod.parse_args(["--max-batches", "1"])

    report = mod.build_probe_report(args)

    assert called is False
    assert report["comparison_available"] is False
    assert report["score_claim"] is False
    assert report["dispatch_attempted"] is False
    assert report["custody_labels"]["score_claim_axis"] == "none"
    assert report["custody_labels"]["contest_cpu_axis_claim"] is False
    assert report["custody_labels"]["contest_cuda_axis_claim"] is False
    assert report["custody_labels"]["dispatch_attempted"] is False
    assert report["device_axis_custody"]["claimed_score_axes"] == []
    assert report["device_axis_custody"]["contest_cpu_axis_claim"] is False
    assert report["device_axis_custody"]["contest_cuda_axis_claim"] is False
    assert report["device_axis_custody"]["dispatch_attempted"] is False
    assert report["local_prerequisite_summary"]["local_host_can_run_full_2x2"] is False
    assert report["local_prerequisite_summary"]["cuda_available"] is False
    assert report["local_prerequisite_summary"]["dali_available"] is False
    assert report["local_prerequisite_summary"]["missing_cuda_dali_prerequisite_codes"] == [
        "cuda_available",
        "dali_available",
        "cuda_dali_runtime_available",
    ]
    remote_contract = report["future_remote_run_contract"]
    assert remote_contract["dispatch_attempted"] is False
    assert remote_contract["requires_dispatch_claim_before_remote_gpu_run"] is True
    assert remote_contract["score_claim"] is False
    assert "--run-forward-cells" in remote_contract["diagnostic_command"]
    assert remote_contract["claim_command_template"][:2] == [
        "tools/claim_lane_dispatch.py",
        "claim",
    ]
    shared_contract = report["shared_input_artifact_contract"]
    assert shared_contract["schema"] == mod.SHARED_INPUT_TENSOR_SCHEMA
    assert shared_contract["requested"] is False
    assert shared_contract["score_claim"] is False
    assert shared_contract["score_claim_valid"] is False
    assert shared_contract["promotion_eligible"] is False
    assert "experiments/dump_scorer_activations.py" in shared_contract["consumer"]
    assert "--shared-input-tensor" in shared_contract["consumer_command_template"]
    assert report["shared_input_artifacts"] == []
    assert report["shared_input_artifact_status"] == "not_requested"

    cells = {cell["cell_id"]: cell for cell in report["intended_cells"]}
    assert list(cells) == [
        "cpu_av",
        "cuda_dali",
        "cuda_av_shared_input",
        "cpu_dali",
    ]
    assert [cell["label"] for cell in report["intended_cells"]] == [
        "CPU+AV",
        "CUDA+DALI",
        "CUDA+AV/shared-input",
        "CPU+DALI",
    ]
    assert cells["cpu_av"]["available"] is True
    assert cells["cpu_av"]["measurement_status"] == "available_not_run"
    assert cells["cuda_dali"]["available"] is False
    assert cells["cuda_av_shared_input"]["available"] is False
    assert cells["cpu_dali"]["available"] is False
    assert "run_forward_cells_false" in cells["cuda_av_shared_input"]["unsupported_codes"]
    assert "cuda_available" in cells["cuda_dali"]["unsupported_codes"]
    assert "run_forward_cells_false" in cells["cpu_dali"]["unsupported_codes"]
    for cell in report["intended_cells"]:
        assert cell["score_claim"] is False
        assert cell["score_claim_valid"] is False
        assert cell["promotion_eligible"] is False
        assert cell["rank_or_kill_eligible"] is False
        assert cell["ready_for_exact_eval_dispatch"] is False
        assert cell["dispatch_attempted"] is False
        assert cell["contest_cpu_axis_claim"] is False
        assert cell["contest_cuda_axis_claim"] is False


def test_dali_runtime_failure_marks_dali_cells_unsupported(monkeypatch) -> None:
    mod = _load_tool()
    monkeypatch.setattr(
        mod,
        "detect_artifacts",
        lambda _args: _artifact_status(
            cuda_available=True,
            dali_available=True,
            cuda_dali_runtime_available=None,
        ),
    )
    monkeypatch.setattr(
        mod,
        "_cuda_dali_available",
        lambda: (False, "nvidia.dali import failed: fixture"),
    )
    args = mod.parse_args(["--max-batches", "1"])

    report = mod.build_probe_report(args)

    assert report["comparison_available"] is False
    assert report["comparison_unavailable_codes"] == ["cuda_dali_runtime_available"]
    cells = {cell["cell_id"]: cell for cell in report["intended_cells"]}
    assert cells["cpu_av"]["available"] is True
    assert cells["cuda_av_shared_input"]["available"] is False
    assert cells["cuda_dali"]["available"] is False
    assert cells["cpu_dali"]["available"] is False
    assert cells["cuda_dali"]["unsupported_codes"] == ["cuda_dali_runtime_available"]
    assert cells["cpu_dali"]["unsupported_codes"] == ["run_forward_cells_false"]
    assert "fixture" in cells["cuda_dali"]["unsupported_reason"]
    assert "run_forward_cells=false" in cells["cpu_dali"]["unsupported_reason"]


def test_raw_cell_availability_does_not_require_scorer_deps_without_forward_cells() -> None:
    mod = _load_tool()
    artifacts = _artifact_status(
        timm_available=False,
        einops_available=False,
        safetensors_torch_available=False,
        segmentation_models_pytorch_available=False,
    )

    raw_cells = {cell["cell_id"]: cell for cell in mod.build_intended_cells(
        artifacts,
        include_forward_requirements=False,
    )}
    forward_cells = {cell["cell_id"]: cell for cell in mod.build_intended_cells(
        artifacts,
        include_forward_requirements=True,
    )}

    assert raw_cells["cpu_av"]["available"] is True
    assert raw_cells["cpu_av"]["forward_requirements_included"] is False
    assert "timm_available" not in raw_cells["cpu_av"]["required_artifacts"]
    assert raw_cells["cuda_av_shared_input"]["available"] is False
    assert raw_cells["cuda_av_shared_input"]["unsupported_codes"] == ["run_forward_cells_false"]
    assert forward_cells["cpu_av"]["available"] is False
    assert forward_cells["cpu_av"]["forward_requirements_included"] is True
    assert "timm_available" in forward_cells["cpu_av"]["unsupported_codes"]


def test_cell_discriminator_plan_names_decoder_and_forward_isolation() -> None:
    mod = _load_tool()
    artifacts = _artifact_status(
        cuda_available=True,
        dali_available=True,
        cuda_dali_runtime_available=True,
    )

    cells = mod.build_intended_cells(artifacts, include_forward_requirements=True)
    plan = {row["comparison_id"]: row for row in mod.build_cell_discriminator_plan(cells)}

    assert all(cell["available"] for cell in cells)
    assert plan["raw_decoder_input_byte_drift_pre_network"]["available"] is True
    assert (
        plan["raw_decoder_input_byte_drift_pre_network"]["isolates"]
        == "decoder_input_byte_drift"
    )
    assert (
        plan["forward_kernel_drift_fixed_pyav_input"]["isolates"]
        == "posenet_segnet_forward_kernel_drift"
    )
    assert (
        plan["forward_kernel_drift_fixed_dali_input"]["isolates"]
        == "posenet_segnet_forward_kernel_drift"
    )
    assert plan["decoder_effect_fixed_cpu_forward"]["score_claim"] is False
    assert plan["decoder_effect_fixed_cuda_forward"]["promotion_eligible"] is False
    assert all(row["dispatch_attempted"] is False for row in plan.values())
    assert all(row["ready_for_exact_eval_dispatch"] is False for row in plan.values())


def test_forward_matrix_summary_distinguishes_plan_from_complete_measurement() -> None:
    mod = _load_tool()
    artifacts = _artifact_status(
        cuda_available=True,
        dali_available=True,
        cuda_dali_runtime_available=True,
    )
    cells = mod.build_intended_cells(artifacts, include_forward_requirements=True)

    not_requested = mod.forward_matrix_summary(
        requested=False,
        intended_cells=cells,
        forward_rows=[],
    )
    complete = mod.forward_matrix_summary(
        requested=True,
        intended_cells=cells,
        forward_rows=[{"batch_order": 0, "forward_matrix_complete": True}],
    )
    incomplete = mod.forward_matrix_summary(
        requested=True,
        intended_cells=cells,
        forward_rows=[{"batch_order": 0, "forward_matrix_complete": False}],
    )

    assert not_requested["complete"] is False
    assert not_requested["status"] == "not_requested"
    assert complete["complete"] is True
    assert complete["status"] == "complete"
    assert incomplete["complete"] is False
    assert incomplete["status"] == "runtime_incomplete"
    assert complete["score_claim"] is False
    assert complete["promotion_eligible"] is False
