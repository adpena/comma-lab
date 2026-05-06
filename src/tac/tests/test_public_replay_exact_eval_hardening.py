from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
REPRO_SCRIPT = REPO_ROOT / "scripts" / "lightning_exact_eval_repro.py"
LAUNCH_SCRIPT = REPO_ROOT / "scripts" / "launch_lightning_batch_job.py"


def _load_script(path: Path, name: str, *, repo_root: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.REPO_ROOT = repo_root
    return module


def _fixture_repo(tmp_path: Path) -> dict[str, Path]:
    archive = tmp_path / "experiments/results/public_pr81/archive.zip"
    archive.parent.mkdir(parents=True)
    archive.write_bytes(b"public pr81 archive bytes")
    baseline = tmp_path / "experiments/results/frontier/contest_auth_eval.json"
    baseline.parent.mkdir(parents=True)
    baseline.write_text(
        json.dumps(
            {
                "score_recomputed_from_components": 0.31453355357318635,
                "archive_size_bytes": 277321,
                "avg_posenet_dist": 0.003,
                "avg_segnet_dist": 0.004,
                "n_samples": 600,
                "provenance": {
                    "device": "cuda",
                    "archive_sha256": "5" * 64,
                    "gpu_t4_match": True,
                },
            }
        )
        + "\n"
    )
    replay = tmp_path / "experiments/results/public_pr81/replay_submission"
    replay.mkdir(parents=True)
    (replay / "inflate.sh").write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
        'python "$HERE/inflate.py" "$1" "$2" "$3"\n'
    )
    (replay / "config.env").write_text("")
    (replay / "inflate.py").write_text(
        "from pathlib import Path\n"
        "Path(__file__).with_name('range_mask_codec.cpp').read_text()\n"
    )
    (replay / "range_mask_codec.cpp").write_text("// range decoder\n")
    return {
        "archive": archive,
        "baseline": baseline,
        "inflate_sh": replay / "inflate.sh",
        "inflate_py": replay / "inflate.py",
        "config_env": replay / "config.env",
        "range_mask_codec": replay / "range_mask_codec.cpp",
    }


def _repo_rel(repo_root: Path, path: Path) -> str:
    return path.relative_to(repo_root).as_posix()


def _flag_value(cmd: list[str], flag: str) -> str:
    return cmd[cmd.index(flag) + 1]


def test_public_replay_repro_plan_forwards_external_inflate_and_stages_runtime(
    tmp_path: Path,
) -> None:
    paths = _fixture_repo(tmp_path)
    module = _load_script(
        REPRO_SCRIPT,
        "lightning_exact_eval_repro_public_replay_test",
        repo_root=tmp_path,
    )

    args = module.build_parser().parse_args(
        [
            "--job-name",
            "public_pr81_t4_replay",
            "--archive",
            _repo_rel(tmp_path, paths["archive"]),
            "--baseline-json",
            _repo_rel(tmp_path, paths["baseline"]),
            "--inflate-sh",
            _repo_rel(tmp_path, paths["inflate_sh"]),
            "--stage-workspace",
            "--remote",
            "lightning-pact",
            "--studio",
            "pact",
            "--predicted-band",
            "0.3",
            "0.6",
            "--regression-threshold",
            "1.0",
        ]
    )

    plan = module.build_plan(args, repo_root=tmp_path)
    queue_cmd = plan["commands"]["queue_exact_eval"]
    assert queue_cmd is not None

    expected_runtime = {
        _repo_rel(tmp_path, paths["inflate_sh"]),
        _repo_rel(tmp_path, paths["config_env"]),
        _repo_rel(tmp_path, paths["inflate_py"]),
        _repo_rel(tmp_path, paths["range_mask_codec"]),
    }
    assert expected_runtime.issubset(set(plan["artifacts"]))
    assert _flag_value(queue_cmd, "--inflate-sh") == _repo_rel(tmp_path, paths["inflate_sh"])
    assert _flag_value(queue_cmd, "--archive").endswith(_repo_rel(tmp_path, paths["archive"]))
    assert "submissions/robust_current/inflate.sh" not in queue_cmd


def test_public_replay_submit_requires_external_inflate_sibling_closure(
    tmp_path: Path,
) -> None:
    paths = _fixture_repo(tmp_path)
    module = _load_script(
        LAUNCH_SCRIPT,
        "launch_lightning_batch_job_public_replay_test",
        repo_root=tmp_path,
    )
    manifest = tmp_path / "source_manifest.json"
    inflate_rel = _repo_rel(tmp_path, paths["inflate_sh"])

    def write_manifest(rels: list[str]) -> None:
        manifest.write_text(json.dumps({"files": [{"path": rel} for rel in rels]}) + "\n")

    args = module.build_parser().parse_args(
        [
            "exact-eval",
            "--job-name",
            "public_pr81_t4_replay",
            "--archive",
            "/repo/experiments/results/public_pr81/archive.zip",
            "--repo-dir",
            "/repo",
            "--upstream-dir",
            "/upstream",
            "--source-manifest",
            str(manifest),
            "--inflate-sh",
            f"/repo/{inflate_rel}",
            "--studio",
            "pact",
            "--machine",
            "g6e.4xlarge",
            "--adjudicate",
            "--baseline-score",
            "0.314",
            "--predicted-band",
            "0.3",
            "0.6",
            "--regression-threshold",
            "1.0",
            "--expected-archive-sha256",
            "a" * 64,
            "--expected-archive-size-bytes",
            "123",
        ]
    )

    write_manifest(
        [
            "experiments/results/public_pr81/archive.zip",
            inflate_rel,
            _repo_rel(tmp_path, paths["config_env"]),
        ]
    )
    with pytest.raises(SystemExit, match="inflate runtime closure"):
        module._validate_exact_eval_submit_inputs(args)

    write_manifest(
        [
            "experiments/results/public_pr81/archive.zip",
            inflate_rel,
            _repo_rel(tmp_path, paths["config_env"]),
            _repo_rel(tmp_path, paths["inflate_py"]),
        ]
    )
    with pytest.raises(SystemExit, match="range_mask_codec\\.cpp"):
        module._validate_exact_eval_submit_inputs(args)

    write_manifest(
        [
            "experiments/results/public_pr81/archive.zip",
            inflate_rel,
            _repo_rel(tmp_path, paths["config_env"]),
            _repo_rel(tmp_path, paths["inflate_py"]),
            _repo_rel(tmp_path, paths["range_mask_codec"]),
        ]
    )
    module._validate_exact_eval_submit_inputs(args)


def test_public_replay_submit_allows_external_inflate_without_config_env(
    tmp_path: Path,
) -> None:
    paths = _fixture_repo(tmp_path)
    paths["config_env"].unlink()
    module = _load_script(
        LAUNCH_SCRIPT,
        "launch_lightning_batch_job_public_replay_no_config_test",
        repo_root=tmp_path,
    )
    manifest = tmp_path / "source_manifest.json"
    inflate_rel = _repo_rel(tmp_path, paths["inflate_sh"])
    manifest.write_text(
        json.dumps(
            {
                "files": [
                    {"path": "experiments/results/public_pr81/archive.zip"},
                    {"path": inflate_rel},
                    {"path": _repo_rel(tmp_path, paths["inflate_py"])},
                    {"path": _repo_rel(tmp_path, paths["range_mask_codec"])},
                ]
            }
        )
        + "\n"
    )

    args = module.build_parser().parse_args(
        [
            "exact-eval",
            "--job-name",
            "public_pr81_t4_replay_no_config",
            "--archive",
            "/repo/experiments/results/public_pr81/archive.zip",
            "--repo-dir",
            "/repo",
            "--upstream-dir",
            "/upstream",
            "--source-manifest",
            str(manifest),
            "--inflate-sh",
            f"/repo/{inflate_rel}",
            "--studio",
            "pact",
            "--machine",
            "g6e.4xlarge",
            "--adjudicate",
            "--baseline-score",
            "0.314",
            "--predicted-band",
            "0.3",
            "0.6",
            "--regression-threshold",
            "1.0",
            "--expected-archive-sha256",
            "a" * 64,
            "--expected-archive-size-bytes",
            "123",
        ]
    )

    module._validate_exact_eval_submit_inputs(args)


def test_public_replay_submit_blocks_source_embedded_payload_loophole(
    tmp_path: Path,
) -> None:
    paths = _fixture_repo(tmp_path)
    paths["archive"].write_bytes(b"x" * 100)
    paths["inflate_py"].write_text(
        "import base64\n"
        f"PAYLOAD = base64.b85decode({('A' * 70000)!r})\n"
        "print(len(PAYLOAD))\n"
    )
    module = _load_script(
        LAUNCH_SCRIPT,
        "launch_lightning_batch_job_public_replay_payload_guard_test",
        repo_root=tmp_path,
    )
    manifest = tmp_path / "source_manifest.json"
    inflate_rel = _repo_rel(tmp_path, paths["inflate_sh"])
    manifest_rels = [
        "experiments/results/public_pr81/archive.zip",
        inflate_rel,
        _repo_rel(tmp_path, paths["config_env"]),
        _repo_rel(tmp_path, paths["inflate_py"]),
        _repo_rel(tmp_path, paths["range_mask_codec"]),
    ]
    manifest.write_text(json.dumps({"files": [{"path": rel} for rel in manifest_rels]}) + "\n")

    base_args = [
        "exact-eval",
        "--job-name",
        "public_replay_payload_guard",
        "--archive",
        "/repo/experiments/results/public_pr81/archive.zip",
        "--repo-dir",
        "/repo",
        "--upstream-dir",
        "/upstream",
        "--source-manifest",
        str(manifest),
        "--inflate-sh",
        f"/repo/{inflate_rel}",
        "--studio",
        "pact",
        "--machine",
        "g6e.4xlarge",
        "--adjudicate",
        "--baseline-score",
        "0.314",
        "--predicted-band",
        "0.3",
        "0.6",
        "--regression-threshold",
        "1.0",
        "--expected-archive-sha256",
        "a" * 64,
        "--expected-archive-size-bytes",
        "100",
    ]
    args = module.build_parser().parse_args(base_args)

    with pytest.raises(SystemExit, match="source-embedded payload"):
        module._validate_exact_eval_submit_inputs(args)

    waived = module.build_parser().parse_args(
        base_args
        + [
            "--allow-source-embedded-payload-runtime-reason",
            "external PR loophole quarantine only",
        ]
    )
    module._validate_exact_eval_submit_inputs(waived)
