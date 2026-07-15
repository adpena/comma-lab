from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = REPO_ROOT / "tools" / "process_tree_liveness.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("process_tree_liveness", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sample(tool, pid: int, ppid: int, command: str):
    return tool.ProcessSample(
        pid=pid,
        ppid=ppid,
        pgid=pid,
        rss_kb=1024,
        command=command,
    )


def test_walks_wrapper_child_and_grandchild_instead_of_declaring_false_dead() -> None:
    tool = _load_tool()
    samples = {
        100: _sample(tool, 100, 1, "bash launch.sh"),
        101: _sample(tool, 101, 100, "python train_levelset.py"),
        102: _sample(tool, 102, 101, "python async_verdict.py"),
        900: _sample(tool, 900, 1, "unrelated"),
    }
    report = tool.build_liveness_report(
        samples, root_pid=100, command_token="train_levelset.py"
    )
    assert report["status"] == "TREE_PRESENT"
    assert report["authority"] == "ROOT_CUSTODIED_PROCESS_TREE"
    assert report["tree_pids"] == [100, 101, 102]
    assert report["alive"] is True


def test_token_fallback_is_alive_but_explicitly_not_root_custody() -> None:
    tool = _load_tool()
    samples = {101: _sample(tool, 101, 1, "python owed16v2 trainer")}
    report = tool.build_liveness_report(
        samples, root_pid=100, command_token="owed16v2"
    )
    assert report["status"] == "TOKEN_MATCH_WITHOUT_ROOT"
    assert report["authority"] == "TOKEN_FALLBACK_NOT_ROOT_CUSTODY"
    assert report["token_match_pids"] == [101]
    assert report["alive"] is True


def test_absence_is_scoped_to_one_sample_not_called_dead() -> None:
    tool = _load_tool()
    report = tool.build_liveness_report({}, root_pid=100, command_token="owed16v2")
    assert report["status"] == "NOT_PRESENT_AT_SAMPLE"
    assert report["verdict_scope"] == "one_local_process_table_sample"
    assert report["alive"] is False
