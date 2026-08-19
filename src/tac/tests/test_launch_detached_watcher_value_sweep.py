# SPDX-License-Identifier: MIT
"""Watcher-config correction must fix a wrong path by VALUE, not only by key name.

``_derive_watcher_config`` repairs the two keys the launcher happens to know --
``pid_file`` and ``log_path``.  A watcher config is free to carry the SAME
drifted path under another key, and ``success_receipts[].path`` is the live one:
when it is present the function hands it back as ``config_declared`` without
ever inspecting the path inside it.

So the pre-fix behaviour was: the launcher proves ``.../stdout.log`` wrong,
rewrites ``log_path`` to ``.../run.log``, announces the supersession -- and
leaves the identical ``.../stdout.log`` sitting in ``success_receipts[0].path``
in the same file.  One wrong value, corrected in one holder, left in the other,
under a banner saying the config had been made effective.

These tests pin the cure: a value proven wrong under any key is corrected
everywhere it appears, and each replacement is recorded as loudly as a key-name
supersession.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL = REPO_ROOT / "tools" / "launch_detached_process.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("launch_detached_process_for_test", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tool():
    return _load_tool()


def _scenario(tmp_path: Path):
    """The 2026-08-16 drift, verbatim in shape.

    The hand-typed config points at a ``launcher/`` pid path and a ``stdout.log``
    the launcher never writes; the launcher itself passes the real paths in argv.
    The wrong log path appears TWICE -- under ``log_path`` and, byte-identical,
    inside ``success_receipts[0].path``.
    """
    run = tmp_path / "run"
    (run / "watchers").mkdir(parents=True)

    wrong_pid = str(run / "launcher" / "child.pid")
    wrong_log = str(run / "stdout.log")
    right_pid = str(run / "child.pid")
    right_log = str(run / "run.log")
    right_receipt = str(run / "safe_run_status.json")

    config = run / "quality_config.json"
    config.write_text(
        json.dumps(
            {
                "pid_file": wrong_pid,
                "log_path": wrong_log,
                "success_receipts": [{"label": "safe_run_status", "path": wrong_log}],
                "probes": [{"name": "tail", "source": wrong_log}],
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    effective_cmd = [
        "python",
        "tools/safe_run.py",
        "--child-pidfile",
        right_pid,
        "--status-receipt",
        right_receipt,
        "--",
        "python",
        "train.py",
    ]
    return {
        "config": config,
        "run": run,
        "effective_cmd": effective_cmd,
        "wrong_pid": wrong_pid,
        "wrong_log": wrong_log,
        "right_pid": right_pid,
        "right_log": right_log,
    }


def test_key_name_correction_alone_leaves_the_same_wrong_value_behind(tool, tmp_path):
    """The defect, stated as an executable control.

    Running only the two key-name corrections -- exactly what the function did
    before the sweep -- repairs ``log_path`` and leaves the byte-identical wrong
    path in ``success_receipts[0].path``.
    """
    s = _scenario(tmp_path)
    config = json.loads(s["config"].read_text(encoding="utf-8"))

    config["pid_file"] = s["right_pid"]  # key-name correction
    config["log_path"] = s["right_log"]  # key-name correction

    assert config["success_receipts"][0]["path"] == s["wrong_log"]
    assert config["probes"][0]["source"] == s["wrong_log"]


def test_a_value_proven_wrong_is_corrected_under_every_key(tool, tmp_path):
    s = _scenario(tmp_path)

    effective, record = tool._derive_watcher_config(
        s["config"],
        kind="quality",
        effective_cmd=s["effective_cmd"],
        resource_budget={"mode": "derived_and_enforced"},
        out=s["run"],
        log_path=Path(s["right_log"]),
    )

    written = json.loads(Path(effective).read_text(encoding="utf-8"))

    # the two key-name corrections still hold ...
    assert written["pid_file"] == s["right_pid"]
    assert written["log_path"] == s["right_log"]

    # ... and the same wrong value no longer survives anywhere else.
    assert written["success_receipts"][0]["path"] == s["right_log"]
    assert written["probes"][0]["source"] == s["right_log"]
    assert s["wrong_log"] not in json.dumps(written)
    assert s["wrong_pid"] not in json.dumps(written)

    # the caller's file is never mutated
    original = json.loads(s["config"].read_text(encoding="utf-8"))
    assert original["log_path"] == s["wrong_log"]


def test_every_value_sweep_is_recorded_as_loudly_as_a_key_name_supersession(tool, tmp_path):
    s = _scenario(tmp_path)

    _, record = tool._derive_watcher_config(
        s["config"],
        kind="quality",
        effective_cmd=s["effective_cmd"],
        resource_budget={"mode": "derived_and_enforced"},
        out=s["run"],
        log_path=Path(s["right_log"]),
    )

    swept = [row for row in record["supersessions"] if row.get("source") == "value_swept"]
    swept_keys = {row["key"] for row in swept}
    assert "success_receipts[0].path" in swept_keys
    assert "probes[0].source" in swept_keys
    for row in swept:
        assert row["declared"] in (s["wrong_log"], s["wrong_pid"])
        assert row["derived"] in (s["right_log"], s["right_pid"])

    # the record must describe the file the watcher will actually read
    assert record["success_receipts"]["value"][0]["path"] == s["right_log"]


def test_sweep_is_inert_when_nothing_was_proven_wrong(tool, tmp_path):
    """No correction, no sweep: a config that already agrees is left alone."""
    s = _scenario(tmp_path)
    run = s["run"]
    good = run / "good_config.json"
    good.write_text(
        json.dumps(
            {
                "pid_file": s["right_pid"],
                "log_path": s["right_log"],
                "success_receipts": [{"label": "safe_run_status", "path": s["right_log"]}],
            },
            indent=1,
        ),
        encoding="utf-8",
    )

    returned, record = tool._derive_watcher_config(
        good,
        kind="quality",
        effective_cmd=s["effective_cmd"],
        resource_budget={"mode": "derived_and_enforced"},
        out=run,
        log_path=Path(s["right_log"]),
    )

    assert returned == good  # unchanged -> the declared file is returned as-is
    assert record["supersessions"] == []
    assert "value_sweep" not in record


def test_sweep_never_rewrites_an_unrelated_value_that_merely_looks_similar(tool, tmp_path):
    """Only EXACT matches of a proven-wrong value are replaced."""
    s = _scenario(tmp_path)
    config = json.loads(s["config"].read_text(encoding="utf-8"))
    neighbour = s["wrong_log"] + ".1"  # a rotated sibling, not the wrong value
    config["rotated"] = neighbour
    config["unrelated"] = "some/other/path.log"
    s["config"].write_text(json.dumps(config, indent=1), encoding="utf-8")

    effective, _ = tool._derive_watcher_config(
        s["config"],
        kind="quality",
        effective_cmd=s["effective_cmd"],
        resource_budget={"mode": "derived_and_enforced"},
        out=s["run"],
        log_path=Path(s["right_log"]),
    )
    written = json.loads(Path(effective).read_text(encoding="utf-8"))

    assert written["rotated"] == neighbour
    assert written["unrelated"] == "some/other/path.log"
