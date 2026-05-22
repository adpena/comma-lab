# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_plan_mlx_parent_contract_closure_blocks_until_recovered(tmp_path: Path) -> None:
    auth_dir = tmp_path / "modal_auth"
    auth_dir.mkdir()
    (auth_dir / "modal_cpu_auth_eval_local_request.json").write_text(
        json.dumps(
            {
                "archive_sha256": "a" * 64,
                "archive_size_bytes": 178_517,
                "scorer_input_cache_tensor_volume_name": "comma-auth-eval-cache-artifacts",
                "scorer_input_cache_tensor_volume_run_id": "unit_tensor_run",
            }
        ),
        encoding="utf-8",
    )
    (auth_dir / "modal_auth_eval_spawn.json").write_text(
        json.dumps({"call_id": "fc-test"}),
        encoding="utf-8",
    )
    reference_cache = tmp_path / "reference_cache"
    reference_cache.mkdir()
    baseline_response = tmp_path / "baseline_response.json"
    baseline_response.write_text("{}", encoding="utf-8")
    baseline_auth = tmp_path / "baseline_auth.json"
    baseline_auth.write_text("{}", encoding="utf-8")
    baseline_window = tmp_path / "baseline_windows" / "window_0000.json"
    baseline_window.parent.mkdir()
    baseline_window.write_text("{}", encoding="utf-8")
    out_json = tmp_path / "plan.json"
    out_md = tmp_path / "plan.md"

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "plan_mlx_parent_contract_closure.py"),
            "--auth-eval-dir",
            str(auth_dir),
            "--reference-cache-dir",
            str(reference_cache),
            "--output-root",
            str(tmp_path / "closure"),
            "--dataset",
            str(tmp_path / "dataset.json"),
            "--baseline-mlx-response",
            str(baseline_response),
            "--baseline-cpu-auth-eval",
            str(baseline_auth),
            "--baseline-window-response",
            str(baseline_window.parent / "*.json"),
            "--json-out",
            str(out_json),
            "--md-out",
            str(out_md),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    stdout = json.loads(completed.stdout)
    plan = json.loads(out_json.read_text(encoding="utf-8"))
    assert stdout["status"] == "blocked"
    assert plan["next_blocker"] == "modal_auth_eval_not_recovered"
    assert plan["score_claim"] is False
    recover = _step(plan, "recover_modal_call")
    prepare_download = _step(plan, "prepare_tensor_download_dir")
    download_manifest = _step(plan, "download_tensor_manifest_json")
    split_windows = _step(plan, "split_parent_windows")
    rebuild_dataset = _step(plan, "build_rebuilt_dataset")
    assert recover["ready"] is True
    assert prepare_download["ready"] is False
    assert download_manifest["ready"] is False
    assert split_windows["requires"] == ["run_parent_response"]
    assert rebuild_dataset["requires"] == ["split_parent_windows"]
    assert str(baseline_window.parent / "*.json") in rebuild_dataset["shell"]
    assert "unit_tensor_run/scorer_input_cache_tensors/manifest.json" in download_manifest["shell"]
    assert "tools/recover_modal_auth_eval.py" in out_md.read_text(encoding="utf-8")


def test_plan_mlx_parent_contract_closure_advances_to_materialization_blocker(
    tmp_path: Path,
) -> None:
    auth_dir = tmp_path / "modal_auth"
    auth_dir.mkdir()
    (auth_dir / "modal_cpu_auth_eval_local_request.json").write_text(
        json.dumps(
            {
                "archive_size_bytes": 178_517,
                "scorer_input_cache_tensor_volume_run_id": "unit_tensor_run",
            }
        ),
        encoding="utf-8",
    )
    (auth_dir / "contest_auth_eval.json").write_text("{}", encoding="utf-8")
    (auth_dir / "scorer_input_cache_tensor_volume_manifest.json").write_text(
        json.dumps({"schema_version": "modal_auth_eval_tensor_volume_manifest.v1"}),
        encoding="utf-8",
    )
    reference_cache = tmp_path / "reference_cache"
    reference_cache.mkdir()
    output_root = tmp_path / "closure"
    downloaded_cache = output_root / "modal_tensor_volume_download" / "scorer_input_cache_tensors"
    downloaded_cache.mkdir(parents=True)
    for name in (
        "manifest.json",
        "pair_indices.npy",
        "posenet_yuv6_pair.npy",
        "segnet_last_rgb.npy",
    ):
        (downloaded_cache / name).write_text("placeholder", encoding="utf-8")
    out_json = tmp_path / "plan.json"

    subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "plan_mlx_parent_contract_closure.py"),
            "--auth-eval-dir",
            str(auth_dir),
            "--reference-cache-dir",
            str(reference_cache),
            "--output-root",
            str(output_root),
            "--dataset",
            str(tmp_path / "dataset.json"),
            "--json-out",
            str(out_json),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    plan = json.loads(out_json.read_text(encoding="utf-8"))
    assert plan["next_blocker"] == "downloaded_tensor_cache_not_materialized"
    assert _step(plan, "download_tensor_manifest_json")["ready"] is True
    assert _step(plan, "materialize_downloaded_cache")["ready"] is True


def test_plan_mlx_parent_contract_closure_uses_rebuilt_dataset_for_bundle(
    tmp_path: Path,
) -> None:
    auth_dir = tmp_path / "modal_auth"
    auth_dir.mkdir()
    (auth_dir / "modal_cpu_auth_eval_local_request.json").write_text(
        json.dumps(
            {
                "archive_size_bytes": 178_517,
                "scorer_input_cache_tensor_volume_run_id": "unit_tensor_run",
            }
        ),
        encoding="utf-8",
    )
    (auth_dir / "contest_auth_eval.json").write_text("{}", encoding="utf-8")
    (auth_dir / "scorer_input_cache_tensor_volume_manifest.json").write_text(
        json.dumps({"schema_version": "modal_auth_eval_tensor_volume_manifest.v1"}),
        encoding="utf-8",
    )
    reference_cache = tmp_path / "reference_cache"
    reference_cache.mkdir()
    output_root = tmp_path / "closure"
    downloaded_cache = output_root / "modal_tensor_volume_download" / "scorer_input_cache_tensors"
    downloaded_cache.mkdir(parents=True)
    for name in (
        "manifest.json",
        "pair_indices.npy",
        "posenet_yuv6_pair.npy",
        "segnet_last_rgb.npy",
    ):
        (downloaded_cache / name).write_text("placeholder", encoding="utf-8")
    (output_root / "candidate_cache").mkdir(parents=True)
    (output_root / "candidate_parent_0000_0600.json").write_text(
        "{}",
        encoding="utf-8",
    )
    (output_root / "candidate_parent_0000_0001.json").write_text(
        "{}",
        encoding="utf-8",
    )
    (output_root / "candidate_windows_index.json").write_text(
        "{}",
        encoding="utf-8",
    )
    rebuilt_dataset = output_root / "candidate_same_axis_window_response_dataset.json"
    rebuilt_dataset.write_text("{}", encoding="utf-8")
    (output_root / "candidate_parent_contract_strict_v1.json").write_text(
        "{}",
        encoding="utf-8",
    )
    legacy_dataset = tmp_path / "legacy_dataset.json"
    legacy_dataset.write_text("{}", encoding="utf-8")
    existing_candidate_window = tmp_path / "existing_candidate_windows" / "window_0000.json"
    existing_candidate_window.parent.mkdir()
    existing_candidate_window.write_text("{}", encoding="utf-8")
    baseline_window = tmp_path / "baseline_windows" / "window_0000.json"
    baseline_window.parent.mkdir()
    baseline_window.write_text("{}", encoding="utf-8")
    out_json = tmp_path / "plan.json"

    subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "plan_mlx_parent_contract_closure.py"),
            "--auth-eval-dir",
            str(auth_dir),
            "--reference-cache-dir",
            str(reference_cache),
            "--output-root",
            str(output_root),
            "--dataset",
            str(legacy_dataset),
            "--max-pairs",
            "1",
            "--existing-candidate-window-response",
            str(existing_candidate_window.parent / "*.json"),
            "--baseline-window-response",
            str(baseline_window.parent / "*.json"),
            "--json-out",
            str(out_json),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    plan = json.loads(out_json.read_text(encoding="utf-8"))
    assert plan["dataset_rebuild_enabled"] is True
    assert plan["legacy_dataset"] == str(legacy_dataset)
    assert plan["dataset_for_contract"] == str(rebuilt_dataset)
    assert plan["existing_candidate_window_responses"] == [
        str(existing_candidate_window.parent / "*.json")
    ]
    assert plan["expected_dataset_row_count"] == 2
    rebuild = _step(plan, "build_rebuilt_dataset")
    assert str(existing_candidate_window.parent / "*.json") in rebuild["shell"]
    assert "--expected-row-count 2" in rebuild["shell"]
    assert "--require-no-skipped" in rebuild["shell"]
    bundle = _step(plan, "build_contract_bundle")
    refresh = _step(plan, "refresh_parent_plan")
    assert str(rebuilt_dataset) in bundle["shell"]
    assert str(rebuilt_dataset) in refresh["shell"]
    assert str(legacy_dataset) not in bundle["shell"]
    assert str(legacy_dataset) not in refresh["shell"]


def test_plan_mlx_parent_contract_closure_calibration_waits_for_parent_response(
    tmp_path: Path,
) -> None:
    auth_dir = tmp_path / "modal_auth"
    auth_dir.mkdir()
    (auth_dir / "modal_cpu_auth_eval_local_request.json").write_text(
        json.dumps(
            {
                "archive_size_bytes": 178_517,
                "scorer_input_cache_tensor_volume_run_id": "unit_tensor_run",
            }
        ),
        encoding="utf-8",
    )
    (auth_dir / "contest_auth_eval.json").write_text("{}", encoding="utf-8")
    (auth_dir / "scorer_input_cache_tensor_volume_manifest.json").write_text(
        json.dumps({"schema_version": "modal_auth_eval_tensor_volume_manifest.v1"}),
        encoding="utf-8",
    )
    reference_cache = tmp_path / "reference_cache"
    reference_cache.mkdir()
    output_root = tmp_path / "closure"
    downloaded_cache = output_root / "modal_tensor_volume_download" / "scorer_input_cache_tensors"
    downloaded_cache.mkdir(parents=True)
    for name in (
        "manifest.json",
        "pair_indices.npy",
        "posenet_yuv6_pair.npy",
        "segnet_last_rgb.npy",
    ):
        (downloaded_cache / name).write_text("placeholder", encoding="utf-8")
    (output_root / "candidate_cache").mkdir(parents=True)
    baseline_response = tmp_path / "baseline_response.json"
    baseline_response.write_text("{}", encoding="utf-8")
    baseline_auth = tmp_path / "baseline_auth.json"
    baseline_auth.write_text("{}", encoding="utf-8")
    out_json = tmp_path / "plan.json"

    subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "plan_mlx_parent_contract_closure.py"),
            "--auth-eval-dir",
            str(auth_dir),
            "--reference-cache-dir",
            str(reference_cache),
            "--output-root",
            str(output_root),
            "--dataset",
            str(tmp_path / "dataset.json"),
            "--baseline-mlx-response",
            str(baseline_response),
            "--baseline-cpu-auth-eval",
            str(baseline_auth),
            "--json-out",
            str(out_json),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    plan = json.loads(out_json.read_text(encoding="utf-8"))
    assert plan["next_blocker"] == "parent_response_missing"
    assert _step(plan, "write_score_calibration_rows")["ready"] is False


def test_plan_mlx_parent_contract_closure_ignores_partial_legacy_dataset_coverage(
    tmp_path: Path,
) -> None:
    auth_dir = _ready_auth_dir(tmp_path)
    reference_cache = tmp_path / "reference_cache"
    reference_cache.mkdir()
    output_root = _closure_root_with_parent(tmp_path)
    legacy_dataset = tmp_path / "legacy_dataset.json"
    legacy_dataset.write_text(
        json.dumps(
            {
                "rows": [
                    {"family": "mlx_decoder_q", "row_id": f"decoderq-{idx}"}
                    for idx in range(300)
                ]
            }
        ),
        encoding="utf-8",
    )
    baseline_dir = tmp_path / "baseline_windows"
    baseline_dir.mkdir()
    for idx in range(300):
        (baseline_dir / f"baseline_{idx:04d}_{idx + 1:04d}.json").write_text(
            "{}",
            encoding="utf-8",
        )
    out_json = tmp_path / "plan.json"

    subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "plan_mlx_parent_contract_closure.py"),
            "--auth-eval-dir",
            str(auth_dir),
            "--reference-cache-dir",
            str(reference_cache),
            "--output-root",
            str(output_root),
            "--dataset",
            str(legacy_dataset),
            "--baseline-window-response",
            str(baseline_dir / "*.json"),
            "--json-out",
            str(out_json),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    plan = json.loads(out_json.read_text(encoding="utf-8"))
    assert plan["baseline_window_existing_count"] == 300
    assert plan["expected_baseline_window_count"] == 600
    assert plan["next_blocker"] == "baseline_window_response_coverage_incomplete"
    assert _step(plan, "build_rebuilt_dataset")["ready"] is False


def test_plan_mlx_parent_contract_closure_blocks_incomplete_baseline_coverage(
    tmp_path: Path,
) -> None:
    auth_dir = _ready_auth_dir(tmp_path)
    reference_cache = tmp_path / "reference_cache"
    reference_cache.mkdir()
    output_root = _closure_root_with_parent(tmp_path)
    legacy_dataset = tmp_path / "legacy_dataset.json"
    legacy_dataset.write_text(
        json.dumps(
            {
                "rows": [
                    {"family": "mlx_decoder_q", "row_id": f"decoderq-{idx}"}
                    for idx in range(600)
                ]
            }
        ),
        encoding="utf-8",
    )
    baseline_dir = tmp_path / "baseline_windows"
    baseline_dir.mkdir()
    for idx in range(300):
        (baseline_dir / f"baseline_{idx:04d}_{idx + 1:04d}.json").write_text(
            "{}",
            encoding="utf-8",
        )
    out_json = tmp_path / "plan.json"

    subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "plan_mlx_parent_contract_closure.py"),
            "--auth-eval-dir",
            str(auth_dir),
            "--reference-cache-dir",
            str(reference_cache),
            "--output-root",
            str(output_root),
            "--dataset",
            str(legacy_dataset),
            "--baseline-window-response",
            str(baseline_dir / "*.json"),
            "--json-out",
            str(out_json),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    plan = json.loads(out_json.read_text(encoding="utf-8"))
    assert plan["baseline_window_existing_count"] == 300
    assert plan["expected_baseline_window_count"] == 600
    assert plan["next_blocker"] == "baseline_window_response_coverage_incomplete"
    assert _step(plan, "build_rebuilt_dataset")["ready"] is False


def _ready_auth_dir(tmp_path: Path) -> Path:
    auth_dir = tmp_path / "modal_auth"
    auth_dir.mkdir()
    (auth_dir / "modal_cpu_auth_eval_local_request.json").write_text(
        json.dumps(
            {
                "archive_size_bytes": 178_517,
                "scorer_input_cache_tensor_volume_run_id": "unit_tensor_run",
            }
        ),
        encoding="utf-8",
    )
    (auth_dir / "contest_auth_eval.json").write_text("{}", encoding="utf-8")
    (auth_dir / "scorer_input_cache_tensor_volume_manifest.json").write_text(
        json.dumps({"schema_version": "modal_auth_eval_tensor_volume_manifest.v1"}),
        encoding="utf-8",
    )
    return auth_dir


def _closure_root_with_parent(tmp_path: Path) -> Path:
    output_root = tmp_path / "closure"
    downloaded_cache = output_root / "modal_tensor_volume_download" / "scorer_input_cache_tensors"
    downloaded_cache.mkdir(parents=True)
    for name in (
        "manifest.json",
        "pair_indices.npy",
        "posenet_yuv6_pair.npy",
        "segnet_last_rgb.npy",
    ):
        (downloaded_cache / name).write_text("placeholder", encoding="utf-8")
    (output_root / "candidate_cache").mkdir(parents=True)
    (output_root / "candidate_parent_0000_0600.json").write_text(
        "{}",
        encoding="utf-8",
    )
    (output_root / "candidate_windows_index.json").write_text(
        "{}",
        encoding="utf-8",
    )
    return output_root


def _step(plan: dict, step_id: str) -> dict:
    for step in plan["steps"]:
        if step["id"] == step_id:
            return step
    raise AssertionError(f"missing step {step_id}")
