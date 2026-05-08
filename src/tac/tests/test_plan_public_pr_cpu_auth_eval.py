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
    assert command[command.index("--inflate-sh") + 1] == row["source"]["key_files"]["inflate_sh"]["path"]


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
