from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from zipfile import ZipFile


def _load_tool():
    repo_root = Path(__file__).resolve().parents[3]
    tool_path = repo_root / "tools" / "plan_dual_device_auth_eval.py"
    spec = importlib.util.spec_from_file_location("plan_dual_device_auth_eval", tool_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {tool_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_dual_plan_emits_cpu_and_cuda_commands_for_same_archive(tmp_path: Path) -> None:
    mod = _load_tool()
    archive = tmp_path / "archive.zip"
    with ZipFile(archive, "w") as zf:
        zf.writestr("x", b"payload")
    inflate = tmp_path / "inflate.sh"
    inflate.write_text("#!/usr/bin/env bash\n", encoding="utf-8")

    plan = mod.build_plan(
        archive=archive,
        inflate_sh=inflate,
        label="fixture",
        repo_root=Path("."),
        run_id="fixture-run",
        output_root=Path("experiments/results/dual_device_auth_eval"),
        upstream_dir=Path("upstream"),
        video_names_file=Path("upstream/public_test_video_names.txt"),
        inflate_timeout=1800,
        evaluate_timeout=1800,
    )

    assert set(plan["evals"]) == {"cuda", "cpu"}
    assert plan["archive"]["path"] == str(archive)
    assert plan["archive"]["sha256"] == mod._sha256(archive)
    for device in ("cuda", "cpu"):
        command = plan["evals"][device]["command"]
        assert command[command.index("--archive") + 1] == str(archive)
        assert command[command.index("--inflate-sh") + 1] == str(inflate)
        assert command[command.index("--device") + 1] == device
    assert plan["evals"]["cuda"]["promotion_eligible_from_this_axis"] is True
    assert plan["evals"]["cpu"]["promotion_eligible_from_this_axis"] is False


def test_public_pr_inputs_resolve_from_wrapped_ledger(tmp_path: Path) -> None:
    mod = _load_tool()
    ledger = tmp_path / "ledger.json"
    ledger.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "pr": 102,
                        "leaderboard_name": "hnerv",
                        "archive": {"path": "a.zip"},
                        "source": {"key_files": {"inflate_sh": {"path": "inflate.sh"}}},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    archive, inflate, label, row = mod._public_pr_inputs(ledger, 102)

    assert archive == Path("a.zip")
    assert inflate == Path("inflate.sh")
    assert label.startswith("public-pr102-hnerv")
    assert row["pr"] == 102
