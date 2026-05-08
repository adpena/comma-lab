from __future__ import annotations

import importlib.util
import json
from pathlib import Path


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
        "upstream_evaluate_py": True,
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
        "cuda_available": True,
        "dali_available": True,
    }
    status.update(overrides)
    return status


def test_matrix_has_four_cells_and_required_nonclaim_fields(monkeypatch) -> None:
    mod = _load_tool("probe_eval_drift_matrix")
    monkeypatch.setattr(mod, "detect_artifacts", lambda _args: _artifact_status())
    args = mod.parse_args([])

    report = mod.build_probe_report(args)

    assert report["schema"] == "eval_drift_matrix_probe.v1"
    assert report["score_claim"] is False
    assert report["promotion_eligible"] is False
    assert report["rank_or_kill_eligible"] is False
    assert report["evidence_grade"] == "diagnostic"
    assert report["matrix_rows"] == ["pyav_av", "dali_nvdec"]
    assert report["matrix_columns"] == ["cpu", "cuda"]
    assert {cell["cell_id"] for cell in report["matrix_cells"]} == {
        "pyav_av__forward_cpu",
        "pyav_av__forward_cuda",
        "dali_nvdec__forward_cpu",
        "dali_nvdec__forward_cuda",
    }
    assert report["fail_closed"] is False


def test_missing_cuda_dali_pyav_fail_closed_without_score_claim(monkeypatch) -> None:
    mod = _load_tool("probe_eval_drift_matrix")
    monkeypatch.setattr(
        mod,
        "detect_artifacts",
        lambda _args: _artifact_status(
            pyav_available=False,
            cuda_available=False,
            dali_available=False,
        ),
    )
    args = mod.parse_args([])

    report = mod.build_probe_report(args)

    assert report["score_claim"] is False
    assert report["promotion_eligible"] is False
    assert report["rank_or_kill_eligible"] is False
    assert report["evidence_grade"] == "diagnostic"
    assert report["fail_closed"] is True
    assert report["fail_closed_reasons"]
    cells = {cell["cell_id"]: cell for cell in report["matrix_cells"]}
    assert cells["pyav_av__forward_cpu"]["available"] is False
    assert "pyav_available=false" in cells["pyav_av__forward_cpu"]["unavailable_reasons"]
    assert cells["dali_nvdec__forward_cuda"]["available"] is False
    assert "cuda_available=false" in cells["dali_nvdec__forward_cuda"]["unavailable_reasons"]
    assert "dali_available=false" in cells["dali_nvdec__forward_cuda"]["unavailable_reasons"]


def test_comparison_plan_separates_row_and_column_drift(monkeypatch) -> None:
    mod = _load_tool("probe_eval_drift_matrix")
    monkeypatch.setattr(mod, "detect_artifacts", lambda _args: _artifact_status())
    args = mod.parse_args([])

    report = mod.build_probe_report(args)
    comparisons = {row["comparison_id"]: row for row in report["comparison_plan"]}

    assert comparisons["decoder_preprocess_drift_fixed_cpu_forward"]["isolates"] == (
        "decoder_plus_preprocess_source_drift"
    )
    assert comparisons["decoder_preprocess_drift_fixed_cpu_forward"]["cell_a"] == (
        "pyav_av__forward_cpu"
    )
    assert comparisons["decoder_preprocess_drift_fixed_cpu_forward"]["cell_b"] == (
        "dali_nvdec__forward_cpu"
    )
    assert comparisons["network_forward_drift_fixed_pyav_decode"]["isolates"] == (
        "preprocess_plus_network_device_drift"
    )
    assert comparisons["network_forward_drift_fixed_pyav_decode"]["cell_a"] == (
        "pyav_av__forward_cpu"
    )
    assert comparisons["network_forward_drift_fixed_pyav_decode"]["cell_b"] == (
        "pyav_av__forward_cuda"
    )
    assert comparisons["raw_decoder_drift_pre_network"]["isolates"] == "decoder_only_raw_rgb_drift"
    assert comparisons["posenet_shared_input_forward_drift"]["isolates"] == (
        "posenet_network_forward_drift_on_shared_input"
    )


def test_sample_video_missing_reason_is_propagated(monkeypatch) -> None:
    mod = _load_tool("probe_eval_drift_matrix")
    monkeypatch.setattr(
        mod,
        "detect_artifacts",
        lambda _args: _artifact_status(
            sample_videos_exist=False,
            sample_video_error="sample video files missing: data/0.mkv",
        ),
    )
    args = mod.parse_args([])

    report = mod.build_probe_report(args)

    assert report["fail_closed"] is True
    assert any("data/0.mkv" in reason for reason in report["fail_closed_reasons"])


def test_main_writes_fail_closed_json_and_returns_two(monkeypatch, tmp_path) -> None:
    mod = _load_tool("probe_eval_drift_matrix")
    monkeypatch.setattr(
        mod,
        "detect_artifacts",
        lambda _args: _artifact_status(cuda_available=False, dali_available=False),
    )
    json_out = tmp_path / "matrix.json"

    rc = mod.main(["--json-out", str(json_out)])

    assert rc == 2
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["fail_closed"] is True
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["rank_or_kill_eligible"] is False
    assert payload["evidence_grade"] == "diagnostic"
