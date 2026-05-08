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


def _artifact_status(**overrides):
    status = {
        "upstream_frame_utils_py": True,
        "video_names_file_exists": True,
        "data_dir_exists": True,
        "sampled_video_names": ["0.mkv"],
        "sampled_video_paths": ["data/0.mkv"],
        "sample_videos_exist": True,
        "sample_video_error": None,
        "pyav_available": True,
        "cuda_available": True,
        "dali_available": True,
    }
    status.update(overrides)
    return status


def test_compare_tensors_reports_lsb_drift_and_nonzero_fraction() -> None:
    mod = _load_tool("probe_eval_loader_drift")
    a = torch.tensor([[[[[0, 10, 20], [30, 40, 50]]]]], dtype=torch.uint8)
    b = torch.tensor([[[[[0, 11, 18], [30, 44, 49]]]]], dtype=torch.uint8)

    comparison = mod.compare_tensors(a, b)

    assert comparison["shape_match"] is True
    assert comparison["max_abs_lsb"] == 4.0
    assert comparison["nonzero_fraction"] == 4 / 6
    assert 1.91 < comparison["rms_abs_lsb"] < 1.92


def test_per_channel_compare_uses_last_rgb_axis() -> None:
    mod = _load_tool("probe_eval_loader_drift")
    a = torch.zeros(1, 1, 1, 2, 3, dtype=torch.uint8)
    b = a.clone()
    b[..., 1] = 7

    rows = mod.per_channel_compare(a, b)

    assert [row["channel"] for row in rows] == [0, 1, 2]
    assert rows[0]["max_abs_lsb"] == 0.0
    assert rows[1]["max_abs_lsb"] == 7.0
    assert rows[2]["max_abs_lsb"] == 0.0


def test_tensor_custody_hashes_shared_input_bytes() -> None:
    mod = _load_tool("probe_eval_loader_drift")
    tensor = torch.tensor([1, 2, 3], dtype=torch.uint8)

    custody = mod.tensor_custody(tensor)

    assert custody["schema"] == mod.SHARED_INPUT_TENSOR_SCHEMA
    assert custody["tensor_role"] == mod.SHARED_INPUT_TENSOR_ROLE
    assert custody["shape"] == [3]
    assert custody["dtype"] == "torch.uint8"
    assert custody["byte_length"] == 3
    assert len(custody["sha256"]) == 64
    assert custody["score_claim"] is False
    assert custody["score_claim_valid"] is False
    assert custody["promotion_eligible"] is False


def test_write_shared_input_tensor_artifact_is_non_promotable(tmp_path: Path) -> None:
    mod = _load_tool("probe_eval_loader_drift")
    tensor = torch.arange(1 * 2 * 3 * 4 * 3, dtype=torch.uint8).reshape(1, 2, 3, 4, 3)

    record = mod.write_shared_input_tensor_artifact(
        output_dir=tmp_path,
        cell_id="cpu_av",
        batch_order=7,
        tensor=tensor,
        video_path="upstream/videos/0.mkv",
        sequence_index=11,
    )

    assert record["schema"] == mod.SHARED_INPUT_TENSOR_SCHEMA
    assert record["tensor_role"] == mod.SHARED_INPUT_TENSOR_ROLE
    assert record["cell_id"] == "cpu_av"
    assert record["score_claim"] is False
    assert record["score_claim_valid"] is False
    assert record["promotion_eligible"] is False
    assert record["rank_or_kill_eligible"] is False
    assert record["ready_for_exact_eval_dispatch"] is False
    assert record["dispatch_attempted"] is False
    assert "tensor" not in record
    assert record["tensor_custody"]["sha256"]
    artifact_path = Path(record["artifact_path"])
    assert artifact_path.exists()
    loaded = torch.load(artifact_path, map_location="cpu", weights_only=True)
    assert loaded["schema"] == mod.SHARED_INPUT_TENSOR_SCHEMA
    assert torch.equal(loaded["tensor"], tensor)
    assert loaded["tensor_custody"]["sha256"] == record["tensor_custody"]["sha256"]
    assert len(record["artifact_file_custody"]["sha256"]) == 64


def test_next_batch_advances_existing_iterator() -> None:
    mod = _load_tool("probe_eval_loader_drift")

    class TinyDataset(torch.utils.data.IterableDataset):
        def __iter__(self):
            yield "a.mp4", 0, torch.zeros(1)
            yield "b.mp4", 1, torch.ones(1)

    iterator = iter(mod._batch_iterator(TinyDataset()))

    first = mod._next_batch(iterator)
    second = mod._next_batch(iterator)

    assert first[0] == "a.mp4"
    assert second[0] == "b.mp4"
    assert second[1] == 1


def test_next_batch_or_none_returns_none_on_exhaustion() -> None:
    mod = _load_tool("probe_eval_loader_drift")
    iterator = iter([])

    assert mod._next_batch_or_none(iterator) is None


def test_default_data_dir_matches_upstream_evaluator() -> None:
    mod = _load_tool("probe_eval_loader_drift")
    args = mod.parse_args([])

    assert args.data_dir == mod.UPSTREAM / "videos"


def test_unavailable_probe_is_non_promotable(monkeypatch) -> None:
    mod = _load_tool("probe_eval_loader_drift")
    monkeypatch.setattr(mod, "detect_artifacts", lambda _args: _artifact_status())
    monkeypatch.setattr(mod, "_cuda_dali_available", lambda: (False, "no cuda fixture"))
    args = mod.parse_args(["--max-batches", "1"])

    report = mod.build_probe_report(args)

    assert report["schema"] == "eval_loader_device_drift_probe.v1"
    assert report["score_axis"] == "diagnostic_loader_drift"
    assert report["comparison_available"] is False
    assert report["comparison_unavailable_reason"] == "no cuda fixture"
    assert report["score_claim"] is False
    assert report["score_claim_valid"] is False
    assert report["promotion_eligible"] is False
    assert report["rank_or_kill_eligible"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    assert report["dispatch_attempted"] is False
    assert report["evidence_grade"] == "diagnostic"
    assert report["diagnostic_kind"] == "loader_drift_probe"
    assert report["loader_device_custody"]["score_path"] == "not_run"
    assert report["loader_device_custody"]["network_forward_device"] == "not_run"
    assert report["loader_device_custody"]["dispatch_attempted"] is False
    assert report["device_axis_custody"]["contest_cuda_claim"] is False
    assert report["device_axis_custody"]["contest_cpu_claim"] is False
    assert report["device_axis_custody"]["score_axis"] == "diagnostic_loader_drift"
    assert report["device_axis_custody"]["dispatch_attempted"] is False
    assert report["future_remote_run_contract"]["dispatch_attempted"] is False
    assert (
        report["future_remote_run_contract"]["requires_dispatch_claim_before_remote_gpu_run"]
        is True
    )
    assert "--run-forward-cells" in report["future_remote_run_contract"]["diagnostic_command"]
    assert report["forward_matrix_complete"] is False
    assert report["forward_matrix_summary"]["status"] == "not_requested"


def test_missing_pyav_fails_closed_before_cuda_probe(monkeypatch) -> None:
    mod = _load_tool("probe_eval_loader_drift")
    monkeypatch.setattr(
        mod,
        "detect_artifacts",
        lambda _args: _artifact_status(pyav_available=False),
    )
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
    assert report["score_claim_valid"] is False
    assert report["promotion_eligible"] is False
    assert report["rank_or_kill_eligible"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    assert report["dispatch_attempted"] is False
    assert "pyav_available=false" in report["comparison_unavailable_reasons"]


def test_missing_sample_video_fails_closed_with_path_reason(monkeypatch) -> None:
    mod = _load_tool("probe_eval_loader_drift")
    monkeypatch.setattr(
        mod,
        "detect_artifacts",
        lambda _args: _artifact_status(
            sample_videos_exist=False,
            sample_video_error="sample video files missing: data/0.mkv",
        ),
    )
    args = mod.parse_args(["--max-batches", "1"])

    report = mod.build_probe_report(args)

    assert report["comparison_available"] is False
    assert any("data/0.mkv" in reason for reason in report["comparison_unavailable_reasons"])
    assert report["score_claim_valid"] is False
