# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _load_tool():
    repo_root = Path(__file__).resolve().parents[3]
    tool_path = repo_root / "tools" / "plan_public_pr_cpu_auth_eval.py"
    spec = importlib.util.spec_from_file_location("plan_public_pr_cpu_auth_eval", tool_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {tool_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_build_plan_forces_cpu_and_non_promotion_semantics() -> None:
    mod = _load_tool()
    row = {
        "pr": 102,
        "leaderboard_name": "hnerv_lc_v2_scale095_rplus1",
        "archive": {
            "path": "experiments/results/public_pr102/archive.zip",
            "bytes": 178981,
            "sha256": "abc",
        },
        "source": {
            "key_files": {
                "inflate_sh": {"path": "experiments/results/public_pr102/source/inflate.sh"}
            }
        },
    }

    plan = mod.build_plan(
        row=row,
        repo_root=Path("."),
        run_id="fixture",
        work_dir=Path("work/cpu"),
        upstream_dir=Path("upstream"),
        video_names_file=Path("upstream/public_test_video_names.txt"),
        inflate_timeout=1800,
        evaluate_timeout=1800,
    )

    assert plan["device"] == "cpu"
    assert plan["promotion_eligible"] is False
    assert plan["score_claim"] is False
    assert plan["rank_or_kill_eligible"] is False
    assert plan["evidence_semantics"] == "public_cpu_leaderboard_reproduction_not_cuda_promotion"
    assert plan["requested_score_axis"] == "contest_cpu"
    assert plan["local_execution_axis"]["score_axis"] in {"contest_cpu", "cpu_advisory"}
    assert plan["local_execution_axis"]["lane_tag"] in {
        "[contest-CPU]",
        "[macOS-CPU advisory]",
        "[CPU advisory]",
    }
    completion = plan["dual_axis_completion"]
    assert completion["paired_score_artifacts_complete"] is False
    assert completion["represented_axes"] == ["contest_cpu"]
    assert completion["missing_axes"] == ["contest_cuda"]
    assert completion["global_priority_eligible"] is False
    assert completion["blockers"] == [
        "missing_contest_cuda_score_artifact",
        "cpu_only_plan_cannot_mark_dual_axis_complete",
    ]
    assert plan["input_closure"]["ready_to_execute"] is False
    assert set(plan["input_closure"]["missing_inputs"]) == {"archive", "inflate_sh"}
    command = plan["command"]
    assert command[command.index("--device") + 1] == "cpu"
    assert command[command.index("--archive") + 1] == row["archive"]["path"]
    assert (
        command[command.index("--inflate-sh") + 1]
        == row["source"]["key_files"]["inflate_sh"]["path"]
    )


def test_local_cpu_host_axis_distinguishes_linux_from_macos() -> None:
    mod = _load_tool()

    linux = mod._local_cpu_host_axis(system="Linux", machine="x86_64")
    macos = mod._local_cpu_host_axis(system="Darwin", machine="arm64")

    assert linux["score_axis"] == "contest_cpu"
    assert linux["lane_tag"] == "[contest-CPU]"
    assert linux["contest_cpu_reproduction_eligible"] is True
    assert linux["hardware_compliance_blocker"] is None
    assert macos["score_axis"] == "cpu_advisory"
    assert macos["lane_tag"] == "[macOS-CPU advisory]"
    assert macos["contest_cpu_reproduction_eligible"] is False
    assert macos["hardware_compliance_blocker"] == "contest_cpu_requires_linux_x86_64"


def test_load_rows_accepts_wrapped_ledger(tmp_path: Path) -> None:
    mod = _load_tool()
    ledger = tmp_path / "ledger.json"
    ledger.write_text(json.dumps({"rows": [{"pr": 100}]}), encoding="utf-8")

    assert mod.load_rows(ledger) == [{"pr": 100}]


def test_public_cpu_execute_refuses_missing_input_closure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod = _load_tool()
    ledger = tmp_path / "ledger.json"
    ledger.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "pr": 102,
                        "leaderboard_name": "hnerv",
                        "archive": {"path": str(tmp_path / "missing.zip")},
                        "source": {
                            "key_files": {
                                "inflate_sh": {"path": str(tmp_path / "missing-inflate.sh")}
                            }
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plan_public_pr_cpu_auth_eval.py",
            "--pr",
            "102",
            "--ledger",
            str(ledger),
            "--execute",
        ],
    )

    with pytest.raises(SystemExit, match="missing inputs: archive, inflate_sh"):
        mod.main()


def test_public_cpu_execute_refuses_advisory_host_even_when_inputs_exist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod = _load_tool()
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"PK\x05\x06" + b"\x00" * 18)
    inflate = tmp_path / "inflate.sh"
    inflate.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    video_names = upstream / "public_test_video_names.txt"
    video_names.write_text("0.mkv\n", encoding="utf-8")
    ledger = tmp_path / "ledger.json"
    ledger.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "pr": 102,
                        "leaderboard_name": "hnerv",
                        "archive": {"path": str(archive)},
                        "source": {
                            "key_files": {
                                "inflate_sh": {"path": str(inflate)}
                            }
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(mod.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(mod.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plan_public_pr_cpu_auth_eval.py",
            "--pr",
            "102",
            "--ledger",
            str(ledger),
            "--upstream-dir",
            str(upstream),
            "--video-names-file",
            str(video_names),
            "--execute",
        ],
    )

    with pytest.raises(SystemExit, match=r"advisory only.*--allow-advisory-host"):
        mod.main()
