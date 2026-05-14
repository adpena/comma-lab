# SPDX-License-Identifier: MIT
"""Tests for credential-safe Lightning exact-eval orchestration."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "lightning_exact_eval_repro.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "lightning_exact_eval_repro_under_test",
        str(SCRIPT),
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixture_repo(tmp_path: Path) -> tuple[Path, Path]:
    archive = tmp_path / "experiments/results/candidate/archive.zip"
    archive.parent.mkdir(parents=True)
    archive.write_bytes(b"candidate archive bytes")
    baseline = tmp_path / "experiments/results/frontier/contest_auth_eval.json"
    baseline.parent.mkdir(parents=True)
    baseline.write_text(
        json.dumps(
            {
                "score_recomputed_from_components": 1.043987524793892,
                "archive_size_bytes": 686635,
                "avg_posenet_dist": 0.00346442,
                "avg_segnet_dist": 0.00400656,
                "n_samples": 600,
                "provenance": {
                    "device": "cuda",
                    "archive_sha256": "0af839abb30e0dfdcfbcbf75247b136db8731196ef26e58374c76a1b562ded7f",
                    "gpu_t4_match": True,
                },
            },
            indent=2,
        )
        + "\n"
    )
    return archive, baseline


def _base_args(archive: Path, baseline: Path) -> list[str]:
    return [
        "--job-name",
        "owv3_exact_eval_test",
        "--archive",
        str(archive),
        "--baseline-json",
        str(baseline),
        "--predicted-band",
        "1.0",
        "1.1",
        "--regression-threshold",
        "1.2",
        "--max-posenet-relative",
        "1.05",
        "--max-segnet-relative",
        "1.002",
        "--studio",
        "pact",
    ]


def _flag_value(cmd: list[str], flag: str) -> str:
    idx = cmd.index(flag)
    return cmd[idx + 1]


def test_stage_command_uses_operator_supplied_ssh_alias_without_key_material(tmp_path: Path) -> None:
    mod = _load_module()
    archive, baseline = _fixture_repo(tmp_path)
    args = mod.build_parser().parse_args(
        [*_base_args(archive, baseline), "--stage-workspace", "--remote", "lightning-pact", "--extra-artifact", str(tmp_path / "experiments/results/candidate/archive.zip")]
    )

    plan = mod.build_plan(args, repo_root=tmp_path)
    stage_cmd = plan["commands"]["stage_workspace"]
    stage_string = plan["command_strings"]["stage_workspace"]

    assert stage_cmd is not None
    assert _flag_value(stage_cmd, "--remote") == "lightning-pact"
    assert "StrictHostKeyChecking" not in stage_string
    assert " -i " not in f" {stage_string} "
    assert "experiments/results/candidate/archive.zip" in plan["artifacts"]
    assert "experiments/results/frontier/contest_auth_eval.json" in plan["artifacts"]


def test_queue_command_is_dry_run_and_uses_writable_remote_workspace_path(tmp_path: Path) -> None:
    mod = _load_module()
    archive, baseline = _fixture_repo(tmp_path)
    args = mod.build_parser().parse_args(_base_args(archive, baseline))

    plan = mod.build_plan(args, repo_root=tmp_path)
    queue_cmd = plan["commands"]["queue_exact_eval"]
    queue_string = plan["command_strings"]["queue_exact_eval"]

    assert queue_cmd is not None
    assert "--dry-run" in queue_cmd
    assert _flag_value(queue_cmd, "--archive") == (
        "/teamspace/studios/this_studio/pact/experiments/results/candidate/archive.zip"
    )
    assert _flag_value(queue_cmd, "--expected-archive-sha256") == hashlib.sha256(
        b"candidate archive bytes"
    ).hexdigest()
    assert _flag_value(queue_cmd, "--expected-archive-size-bytes") == str(len(b"candidate archive bytes"))
    assert "/teamspace/jobs/" not in queue_string
    assert "--adjudicate" in queue_cmd
    assert "--device cuda" not in queue_cmd


def test_external_inflate_runtime_artifacts_include_nested_helpers(tmp_path: Path) -> None:
    mod = _load_module()
    archive, baseline = _fixture_repo(tmp_path)
    runtime = tmp_path / "experiments/results/candidate/submission_dir"
    (runtime / "src").mkdir(parents=True)
    (runtime / "inflate.sh").write_text("#!/usr/bin/env bash\n")
    (runtime / "inflate.py").write_text("from src.model import X\n")
    (runtime / "src" / "model.py").write_text("X = 1\n")
    (runtime / "src" / "codec.py").write_text("Y = 2\n")
    (runtime / "__pycache__").mkdir()
    (runtime / "__pycache__" / "inflate.cpython.pyc").write_bytes(b"cache")

    args = mod.build_parser().parse_args(
        [*_base_args(archive, baseline), "--inflate-sh", str(runtime / "inflate.sh"), "--stage-workspace", "--remote", "lightning-pact"]
    )

    plan = mod.build_plan(args, repo_root=tmp_path)

    assert "experiments/results/candidate/submission_dir/inflate.sh" in plan["artifacts"]
    assert "experiments/results/candidate/submission_dir/inflate.py" in plan["artifacts"]
    assert "experiments/results/candidate/submission_dir/src/model.py" in plan["artifacts"]
    assert "experiments/results/candidate/submission_dir/src/codec.py" in plan["artifacts"]
    assert not any("__pycache__" in item for item in plan["artifacts"])


def test_declared_pact_runtime_dependency_root_is_staged(tmp_path: Path) -> None:
    mod = _load_module()
    archive, baseline = _fixture_repo(tmp_path)
    adapter = tmp_path / "experiments/public_runtime_adapters/pr104/inflate.sh"
    adapter.parent.mkdir(parents=True)
    runtime = tmp_path / "experiments/results/public_pr104/source/submissions/qhnerv_ft_best"
    (runtime / "src").mkdir(parents=True)
    adapter.write_text(
        "#!/usr/bin/env bash\n"
        "# PACT_RUNTIME_DEPENDENCY_ROOT=experiments/results/public_pr104/source/submissions/qhnerv_ft_best\n"
    )
    (runtime / "inflate.py").write_text("from src.model import HNeRVDecoder\n")
    (runtime / "src" / "model.py").write_text("class HNeRVDecoder: pass\n")
    (runtime / "src" / "codec.py").write_text("def parse_archive(): pass\n")
    (runtime / "__pycache__").mkdir()
    (runtime / "__pycache__" / "inflate.cpython.pyc").write_bytes(b"cache")

    args = mod.build_parser().parse_args(
        [
            *_base_args(archive, baseline),
            "--inflate-sh",
            str(adapter),
            "--stage-workspace",
            "--remote",
            "lightning-pact",
        ]
    )

    plan = mod.build_plan(args, repo_root=tmp_path)

    root_rel = "experiments/results/public_pr104/source/submissions/qhnerv_ft_best"
    assert plan["inflate_runtime"]["declared_dependency_roots"] == [root_rel]
    assert f"{root_rel}/inflate.py" in plan["artifacts"]
    assert f"{root_rel}/src/model.py" in plan["artifacts"]
    assert f"{root_rel}/src/codec.py" in plan["artifacts"]
    assert not any("__pycache__" in item for item in plan["artifacts"])


def test_baseline_json_populates_adjudication_flags(tmp_path: Path) -> None:
    mod = _load_module()
    archive, baseline = _fixture_repo(tmp_path)
    args = mod.build_parser().parse_args(_base_args(archive, baseline))

    plan = mod.build_plan(args, repo_root=tmp_path)
    queue_cmd = plan["commands"]["queue_exact_eval"]
    assert queue_cmd is not None

    assert _flag_value(queue_cmd, "--baseline-score") == "1.0439875247938919"
    assert _flag_value(queue_cmd, "--baseline-archive-bytes") == "686635"
    assert _flag_value(queue_cmd, "--baseline-posenet-dist") == "0.00346442"
    assert _flag_value(queue_cmd, "--baseline-segnet-dist") == "0.00400656"
    assert _flag_value(queue_cmd, "--component-reference-label") == (
        "experiments/results/frontier/contest_auth_eval.json"
    )
    assert plan["queue_metadata"]["baseline_json"] == "experiments/results/frontier/contest_auth_eval.json"


def test_submit_requires_staging_or_explicit_remote_custody_and_target_backend(tmp_path: Path) -> None:
    mod = _load_module()
    archive, baseline = _fixture_repo(tmp_path)

    no_stage = mod.build_parser().parse_args([*_base_args(archive, baseline), "--submit"])
    with pytest.raises(ValueError, match="--stage-workspace or --allow-unstaged-submit"):
        mod.build_plan(no_stage, repo_root=tmp_path)

    no_backend_args = _base_args(archive, baseline)
    studio_idx = no_backend_args.index("--studio")
    del no_backend_args[studio_idx : studio_idx + 2]
    no_backend = mod.build_parser().parse_args(
        [*no_backend_args, "--submit", "--allow-unstaged-submit"]
    )
    with pytest.raises(ValueError, match="--studio or --image"):
        mod.build_plan(no_backend, repo_root=tmp_path)


def test_submit_requires_remote_alias_before_forwarding_submit_preflight(tmp_path: Path) -> None:
    mod = _load_module()
    archive, baseline = _fixture_repo(tmp_path)
    args = mod.build_parser().parse_args(
        [*_base_args(archive, baseline), "--submit", "--allow-unstaged-submit"]
    )
    args.remote = None

    with pytest.raises(ValueError, match="--remote"):
        mod.build_plan(args, repo_root=tmp_path)


def test_stage_workspace_rejects_bare_lightning_ssh_host(tmp_path: Path) -> None:
    mod = _load_module()
    archive, baseline = _fixture_repo(tmp_path)
    args = mod.build_parser().parse_args(
        [*_base_args(archive, baseline), "--stage-workspace", "--remote", "ssh.lightning.ai"]
    )

    with pytest.raises(ValueError, match="bare ssh\\.lightning\\.ai"):
        mod.build_plan(args, repo_root=tmp_path)


def test_wrapper_argparse_rejects_misspelled_remote_flag_with_real_surface(
    capsys: pytest.CaptureFixture[str],
) -> None:
    mod = _load_module()

    with pytest.raises(SystemExit):
        mod.build_parser().parse_args(["--job-name", "x", "--archive", "a.zip", "--rmote", "lightning-pact"])

    captured = capsys.readouterr()
    assert "--rmote: --remote" in captured.err
    assert "Known options include:" in captured.err


def test_submit_forwards_dispatch_claim_guard_flags(tmp_path: Path) -> None:
    mod = _load_module()
    archive, baseline = _fixture_repo(tmp_path)
    args = mod.build_parser().parse_args(
        [*_base_args(archive, baseline), "--submit", "--stage-workspace", "--remote", "lightning-pact", "--dispatch-lane-id", "lane_renderer_eval", "--dispatch-claims-path", ".omx/state/active_lane_dispatch_claims.md"]
    )

    plan = mod.build_plan(args, repo_root=tmp_path)
    queue_cmd = plan["commands"]["queue_exact_eval"]
    assert queue_cmd is not None
    assert "--dry-run" not in queue_cmd
    assert _flag_value(queue_cmd, "--dispatch-lane-id") == "lane_renderer_eval"
    assert _flag_value(queue_cmd, "--dispatch-claims-path") == ".omx/state/active_lane_dispatch_claims.md"


def test_queue_command_forwards_exact_eval_env_overrides(tmp_path: Path) -> None:
    mod = _load_module()
    archive, baseline = _fixture_repo(tmp_path)
    args = mod.build_parser().parse_args(
        [
            *_base_args(archive, baseline),
            "--env",
            "INFLATE_TORCH_SPEC=torch==2.5.1+cu124",
            "--env",
            "INFLATE_TORCHVISION_SPEC=torchvision==0.20.1+cu124",
            "--env",
            "UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu124",
        ]
    )

    plan = mod.build_plan(args, repo_root=tmp_path)
    queue_cmd = plan["commands"]["queue_exact_eval"]
    assert queue_cmd is not None

    env_values = [
        queue_cmd[index + 1]
        for index, value in enumerate(queue_cmd)
        if value == "--env"
    ]
    assert env_values == [
        "INFLATE_TORCH_SPEC=torch==2.5.1+cu124",
        "INFLATE_TORCHVISION_SPEC=torchvision==0.20.1+cu124",
        "UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu124",
    ]
    assert plan["env"] == env_values


def test_queue_command_can_request_component_trace(tmp_path: Path) -> None:
    mod = _load_module()
    archive, baseline = _fixture_repo(tmp_path)
    args = mod.build_parser().parse_args(
        [*_base_args(archive, baseline), "--component-trace", "--component-trace-top-k", "96"]
    )

    plan = mod.build_plan(args, repo_root=tmp_path)
    queue_cmd = plan["commands"]["queue_exact_eval"]
    assert queue_cmd is not None

    assert "--component-trace" in queue_cmd
    assert _flag_value(queue_cmd, "--component-trace-top-k") == "96"
    assert plan["component_trace"] is True
    assert plan["component_trace_top_k"] == 96


def test_submit_forwards_dispatch_claim_break_glass_reason(tmp_path: Path) -> None:
    mod = _load_module()
    archive, baseline = _fixture_repo(tmp_path)
    args = mod.build_parser().parse_args(
        [*_base_args(archive, baseline), "--submit", "--stage-workspace", "--remote", "lightning-pact", "--allow-missing-dispatch-claim-reason", "operator reviewed emergency rerun"]
    )

    plan = mod.build_plan(args, repo_root=tmp_path)
    queue_cmd = plan["commands"]["queue_exact_eval"]
    assert queue_cmd is not None
    assert _flag_value(queue_cmd, "--allow-missing-dispatch-claim-reason") == (
        "operator reviewed emergency rerun"
    )


def test_read_only_sdk_artifact_view_is_rejected_as_output_dir(tmp_path: Path) -> None:
    mod = _load_module()
    archive, baseline = _fixture_repo(tmp_path)
    args = mod.build_parser().parse_args(
        [*_base_args(archive, baseline), "--output-dir", "/teamspace/jobs/owv3/artifacts"]
    )
    with pytest.raises(ValueError, match="read-only artifact view"):
        mod.build_plan(args, repo_root=tmp_path)


def test_submit_consistency_guard_rejects_file_drift_after_stage_manifest(tmp_path: Path) -> None:
    mod = _load_module()
    archive, baseline = _fixture_repo(tmp_path)
    inflate = tmp_path / "submissions/robust_current/inflate.sh"
    inflate.parent.mkdir(parents=True)
    inflate.write_text("#!/usr/bin/env bash\n")
    args = mod.build_parser().parse_args(
        [*_base_args(archive, baseline), "--stage-workspace", "--remote", "lightning-pact"]
    )
    plan = mod.build_plan(args, repo_root=tmp_path)
    manifest_path = tmp_path / plan["paths"]["manifest_out"]
    manifest_path.parent.mkdir(parents=True)
    files = []
    for rel in plan["artifacts"]:
        path = tmp_path / rel
        files.append(
            {
                "path": rel,
                "role": "artifact",
                "bytes": path.stat().st_size,
                "sha256": mod.sha256_file(path),
            }
        )
    manifest_path.write_text(json.dumps({"schema_version": 1, "files": files}, indent=2) + "\n")

    archive.write_bytes(b"candidate archive bytes drifted")

    with pytest.raises(ValueError, match="changed after staging source manifest"):
        mod._validate_staged_manifest_consistency(plan, repo_root=tmp_path)


def test_public_replay_preflight_must_match_current_adapter_sha(tmp_path: Path) -> None:
    mod = _load_module()
    archive, baseline = _fixture_repo(tmp_path)
    inflate = tmp_path / "experiments/public_runtime_adapters/pr106/inflate.sh"
    inflate.parent.mkdir(parents=True)
    inflate.write_text("#!/usr/bin/env bash\n")
    preflight = tmp_path / "experiments/results/pr106/public_replay_preflight.json"
    preflight.parent.mkdir(parents=True, exist_ok=True)
    preflight.write_text(
        json.dumps(
            {
                "ready_for_exact_eval_dispatch": True,
                "runtime": {
                    "inflate_sh": "experiments/public_runtime_adapters/pr106/inflate.sh",
                    "inflate_sh_sha256": "stale",
                    "runtime_manifest": {},
                },
            }
        )
        + "\n"
    )
    args = mod.build_parser().parse_args(
        [*_base_args(archive, baseline), "--inflate-sh", str(inflate), "--queue-metadata", "public_preflight=experiments/results/pr106/public_replay_preflight.json"]
    )
    plan = mod.build_plan(args, repo_root=tmp_path)

    with pytest.raises(ValueError, match="inflate_sh_sha256 is stale"):
        mod._validate_public_replay_preflight(plan, repo_root=tmp_path)


def test_public_replay_preflight_requires_external_roots_for_declared_dependency(tmp_path: Path) -> None:
    mod = _load_module()
    archive, baseline = _fixture_repo(tmp_path)
    inflate = tmp_path / "experiments/public_runtime_adapters/pr106/inflate.sh"
    inflate.parent.mkdir(parents=True)
    runtime = tmp_path / "experiments/results/pr106/source"
    runtime.mkdir(parents=True)
    (runtime / "inflate.py").write_text("print('runtime')\n")
    inflate.write_text(
        "#!/usr/bin/env bash\n"
        "export PACT_RUNTIME_DEPENDENCY_ROOT=experiments/results/pr106/source\n"
    )
    preflight = tmp_path / "experiments/results/pr106/public_replay_preflight.json"
    preflight.parent.mkdir(parents=True, exist_ok=True)
    preflight.write_text(
        json.dumps(
            {
                "ready_for_exact_eval_dispatch": True,
                "runtime": {
                    "inflate_sh": "experiments/public_runtime_adapters/pr106/inflate.sh",
                    "inflate_sh_sha256": mod.sha256_file(inflate),
                    "runtime_manifest": {"external_dependency_roots": []},
                },
            }
        )
        + "\n"
    )
    args = mod.build_parser().parse_args(
        [*_base_args(archive, baseline), "--inflate-sh", str(inflate), "--queue-metadata", "public_preflight=experiments/results/pr106/public_replay_preflight.json"]
    )
    plan = mod.build_plan(args, repo_root=tmp_path)

    with pytest.raises(ValueError, match="no external_dependency_roots"):
        mod._validate_public_replay_preflight(plan, repo_root=tmp_path)
