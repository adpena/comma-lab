from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = REPO_ROOT / "tools" / "build_kaggle_proxy_sweep_kernel.py"


def load_tool():
    spec = importlib.util.spec_from_file_location(
        "build_kaggle_proxy_sweep_kernel_under_test",
        TOOL_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_builds_private_gpu_kernel_and_proxy_contract(tmp_path: Path) -> None:
    tool = load_tool()
    kernel_dir = tmp_path / "kernel"

    result = tool.build_kernel(
        kernel_dir=kernel_dir,
        owner="adpena",
        slug="unit-proxy-sweep",
        title="Unit Proxy Sweep",
    )

    metadata = json.loads((kernel_dir / "kernel-metadata.json").read_text())
    manifest = json.loads((kernel_dir / "proxy_sweep_build_manifest.json").read_text())

    assert metadata["id"] == "adpena/unit-proxy-sweep"
    assert metadata["kernel_type"] == "script"
    assert metadata["is_private"] is True
    assert metadata["enable_gpu"] is True
    assert metadata["code_file"] == "pr101_proxy_sweep.py"
    assert manifest["score_claim"] is False
    assert manifest["score_claim_valid"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["proxy_only"] is True
    assert manifest["dispatch_claim_required"] is True
    assert manifest["lane_id"] == "kaggle_pr101_proxy_sweep"
    assert manifest["claim_command_dry_run"][:4] == [
        ".venv/bin/python",
        "tools/claim_lane_dispatch.py",
        "claim",
        "--dry-run",
    ]
    assert manifest["claim_command"][:3] == [
        ".venv/bin/python",
        "tools/claim_lane_dispatch.py",
        "claim",
    ]
    assert "--dry-run" not in manifest["claim_command"]
    assert manifest["claim_command"][manifest["claim_command"].index("--platform") + 1] == "kaggle"
    assert (
        manifest["claim_command"][manifest["claim_command"].index("--status") + 1]
        == "active_proxy_dispatch"
    )
    assert manifest["safe_push_sequence"] == [
        manifest["claim_command_dry_run"],
        manifest["claim_command"],
        manifest["push_command"],
    ]
    assert manifest["safe_push_sequence_text"] == [
        manifest["claim_command_dry_run_text"],
        manifest["claim_command_text"],
        manifest["push_command_text"],
    ]
    assert "score_claim=false" in manifest["claim_command_text"]
    assert "'Kaggle PR101 proxy sweep only;" in manifest["claim_command_text"]
    assert "--force" in manifest["terminal_claim_command_template"]
    assert manifest["exact_auth_eval_performed"] is False
    assert manifest["archive_zip_emitted"] is False
    assert manifest["inflate_runtime_emitted"] is False
    assert manifest["evidence_semantics"] == (
        "kaggle_gpu_proxy_config_search_only_not_exact_auth_eval"
    )
    assert result.push_command == [
        "uv",
        "run",
        "--with",
        "kaggle",
        "kaggle",
        "kernels",
        "push",
        "-p",
        str(kernel_dir),
    ]
    assert result.claim_command == manifest["claim_command"]
    assert result.claim_dry_run_command == manifest["claim_command_dry_run"]


def test_builds_bias_refine_profile_with_runtime_consumed_param_schema(tmp_path: Path) -> None:
    tool = load_tool()
    kernel_dir = tmp_path / "bias_kernel"

    result = tool.build_kernel(
        profile_name="pr101_bias_refine",
        kernel_dir=kernel_dir,
        owner="adpena",
        agent="codex:test",
    )

    metadata = json.loads((kernel_dir / "kernel-metadata.json").read_text())
    manifest = json.loads((kernel_dir / "proxy_sweep_build_manifest.json").read_text())

    assert metadata["id"] == "adpena/pr101-bias-refine"
    assert metadata["code_file"] == "pr101_bias_refine.py"
    assert result.script_path.name == "pr101_bias_refine.py"
    assert manifest["profile"] == "pr101_bias_refine"
    assert manifest["lane_id"] == "kaggle_pr101_bias_refine"
    assert manifest["lane_class"] == "pr101_kaggle_bias_refine"
    assert manifest["candidate_family"] == "pr101_runtime_consumed_bias_refinement"
    assert manifest["param_schema"] == "pr101_kaggle_proxy_bias_runtime_params_v1"
    assert "all params are runtime-consumed" in manifest["claim_command_text"]
    assert manifest["score_claim"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False


def test_cli_prints_push_command_without_launching(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "kernel"

    proc = subprocess.run(
        [
            sys.executable,
            str(TOOL_PATH),
            "--kernel-dir",
            str(kernel_dir),
            "--owner",
            "adpena",
            "--slug",
            "unit-proxy-sweep",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Claim dry-run command:" in proc.stdout
    assert "Claim command:" in proc.stdout
    assert "Operator-controlled launch command after successful claim:" in proc.stdout
    assert "tools/claim_lane_dispatch.py claim --dry-run" in proc.stdout
    assert "tools/claim_lane_dispatch.py claim --lane-id kaggle_pr101_proxy_sweep" in proc.stdout
    assert "'Kaggle PR101 proxy sweep only;" in proc.stdout
    assert f"uv run --with kaggle kaggle kernels push -p {kernel_dir}" in proc.stdout
    assert "kaggle kernels push" in proc.stdout
    assert "score_claim" in proc.stdout
    assert '"score_claim": false' in proc.stdout
    assert "subprocess.run" not in proc.stdout


def test_generated_kernel_runs_and_writes_no_score_claim_outputs(tmp_path: Path) -> None:
    tool = load_tool()
    kernel_dir = tmp_path / "kernel"
    output_dir = tmp_path / "run"
    tool.build_kernel(kernel_dir=kernel_dir)

    proc = subprocess.run(
        [
            sys.executable,
            str(kernel_dir / "pr101_proxy_sweep.py"),
            "--optimizer",
            "random",
            "--max-trials",
            "3",
            "--seed",
            "7",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    stdout_payload = json.loads(proc.stdout)
    manifest = json.loads((output_dir / "proxy_sweep_manifest.json").read_text())
    rows = [
        json.loads(line)
        for line in (output_dir / "proxy_sweep_results.jsonl").read_text().splitlines()
    ]

    assert stdout_payload["score_claim"] is False
    assert stdout_payload["ready_for_exact_eval_dispatch"] is False
    assert manifest["score_claim"] is False
    assert manifest["score_claim_valid"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["proxy_only"] is True
    assert manifest["exact_auth_eval_performed"] is False
    assert manifest["contest_cuda_auth_eval"] is False
    assert manifest["mps_auth_eval"] is False
    assert manifest["archive_zip_emitted"] is False
    assert manifest["inflate_runtime_emitted"] is False
    assert len(rows) == 3
    assert all(row["score_claim"] is False for row in rows)
    assert all(row["ready_for_exact_eval_dispatch"] is False for row in rows)
    assert all(row["proxy_only"] is True for row in rows)


def test_generated_bias_refine_kernel_emits_only_runtime_consumed_bias_params(
    tmp_path: Path,
) -> None:
    tool = load_tool()
    kernel_dir = tmp_path / "kernel"
    output_dir = tmp_path / "run"
    tool.build_kernel(profile_name="pr101_bias_refine", kernel_dir=kernel_dir)

    proc = subprocess.run(
        [
            sys.executable,
            str(kernel_dir / "pr101_bias_refine.py"),
            "--optimizer",
            "random",
            "--max-trials",
            "4",
            "--seed",
            "11",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    stdout_payload = json.loads(proc.stdout)
    manifest = json.loads((output_dir / "proxy_sweep_manifest.json").read_text())
    rows = [
        json.loads(line)
        for line in (output_dir / "proxy_sweep_results.jsonl").read_text().splitlines()
    ]

    assert stdout_payload["profile"] == "pr101_bias_refine"
    assert stdout_payload["param_schema"] == "pr101_kaggle_proxy_bias_runtime_params_v1"
    assert manifest["profile"] == "pr101_bias_refine"
    assert manifest["lane_id"] == "kaggle_pr101_bias_refine"
    assert manifest["param_schema"] == "pr101_kaggle_proxy_bias_runtime_params_v1"
    assert set(manifest["search_space"]) == {"bias_b", "bias_g", "bias_r"}
    assert len(rows) == 4
    assert all(set(row["params"]) == {"bias_b", "bias_g", "bias_r"} for row in rows)
    assert all(row["param_schema"] == "pr101_kaggle_proxy_bias_runtime_params_v1" for row in rows)
    assert all(row["score_claim"] is False for row in rows)
    assert all(row["ready_for_exact_eval_dispatch"] is False for row in rows)


def test_generated_kernel_source_has_no_exact_eval_or_mps_auth_path(tmp_path: Path) -> None:
    tool = load_tool()
    kernel_dir = tmp_path / "kernel"
    tool.build_kernel(kernel_dir=kernel_dir)

    source = (kernel_dir / "pr101_proxy_sweep.py").read_text()
    lowered = source.lower()

    assert "contest_auth_eval.py" not in source
    assert "--device mps" not in lowered
    assert '"score_claim": true' not in lowered
    assert '"ready_for_exact_eval_dispatch": true' not in lowered
    assert "archive.zip" in source
    assert "no archive.zip is emitted" in source
