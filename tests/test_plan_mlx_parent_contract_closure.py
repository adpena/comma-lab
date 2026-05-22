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
    download = _step(plan, "download_tensor_volume")
    assert recover["ready"] is True
    assert download["ready"] is False
    assert "unit_tensor_run/" in download["shell"]
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
    (output_root / "modal_tensor_volume_download" / "scorer_input_cache_tensors").mkdir(
        parents=True
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
    assert _step(plan, "download_tensor_volume")["ready"] is True
    assert _step(plan, "materialize_downloaded_cache")["ready"] is True


def _step(plan: dict, step_id: str) -> dict:
    for step in plan["steps"]:
        if step["id"] == step_id:
            return step
    raise AssertionError(f"missing step {step_id}")
