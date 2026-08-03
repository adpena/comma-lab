# SPDX-License-Identifier: MIT
"""Tests for the ``findings`` field on tools/subagent_checkpoint.py.

Arm ``ddm_rs2``, 2026-08-03. Bug class: the checkpoint store carried
``step``/``status``/``files_touched``/``next_action``/``notes`` and answered
exactly one question — *"where do I resume"*. It never answered *"what did we
learn"*. When four arms were killed mid-flight by a provider weekly usage limit
on 2026-08-03, ``ddm_gd2``'s checkpoint said only "verify mt1 rate ladder;
determine whether a no-train ds=32 archive yields a meaningful d_seg" — a next
action, not a finding. Its structural blocker was not lost: sister arm
``ddm_gd3`` spent a whole unit RE-DERIVING it (``db3abc5b4a``). The bug class
is PAID REDISCOVERY, which is the true and cheaper claim.

The fix is additive and legacy-compatible per the ``lane_id`` /
``respawn_context`` / ``expected_outputs`` precedent: old rows must still load.
These tests pin (a) the field round-trips, (b) validation refuses malformed
input, (c) legacy rows without the field still load through every reader,
(d) the CLI ``--finding`` flag is repeatable and does NOT comma-split (findings
are prose and prose contains commas), and (e) the ``read --findings``
knowledge-log query.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SUBAGENT_CHECKPOINT_PATH = REPO_ROOT / "tools" / "subagent_checkpoint.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "subagent_checkpoint_findings", SUBAGENT_CHECKPOINT_PATH
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def mod(tmp_path, monkeypatch):
    module = _load_module()
    state_dir = tmp_path / ".omx" / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(module, "STATE_DIR", state_dir)
    monkeypatch.setattr(module, "JSONL_PATH", state_dir / "subagent_progress.jsonl")
    monkeypatch.setattr(module, "LOCK_PATH", state_dir / ".subagent_progress.lock")
    return module


def _write(mod, **kw):
    base = {
        "subagent_id": "ddm_test",
        "step": 1,
        "status": "in_progress",
        "files_touched": [],
        "next_action": "next",
    }
    base.update(kw)
    return mod.append_checkpoint(**base)


# ─── (a) round-trip ──────────────────────────────────────────────────────


def test_findings_round_trip(mod):
    _write(mod, findings=["the probe counted its own watchers"])
    rec = mod.latest_checkpoint("ddm_test")
    assert rec["findings"] == ["the probe counted its own watchers"]


def test_multiple_findings_preserved_in_order(mod):
    got = _write(mod, findings=["first", "second", "third"])
    assert got["findings"] == ["first", "second", "third"]
    assert mod.latest_checkpoint("ddm_test")["findings"] == [
        "first",
        "second",
        "third",
    ]


def test_findings_default_is_none_not_empty_list(mod):
    """None (never recorded) must stay distinguishable from [] (recorded none)."""
    _write(mod)
    assert mod.latest_checkpoint("ddm_test")["findings"] is None
    _write(mod, step=2, findings=[])
    assert mod.read_checkpoints("ddm_test")[-1]["findings"] == []


def test_findings_list_is_copied_not_aliased(mod):
    caller = ["mutable"]
    _write(mod, findings=caller)
    caller.append("added after the write")
    assert mod.latest_checkpoint("ddm_test")["findings"] == ["mutable"]


def test_finding_with_commas_is_one_finding(mod):
    """The whole point of --finding over a comma-split flag."""
    prose = "prefix n=73, its own n600 said +0.152, so a prefix is a different population"
    _write(mod, findings=[prose])
    assert mod.latest_checkpoint("ddm_test")["findings"] == [prose]


def test_findings_survive_json_serialization(mod):
    _write(mod, findings=["unicode: ΔS −0.126923 · 45×"])
    raw = mod.JSONL_PATH.read_text().strip()
    assert json.loads(raw)["findings"] == ["unicode: ΔS −0.126923 · 45×"]


# ─── (b) validation ──────────────────────────────────────────────────────


def test_findings_rejects_bare_string(mod):
    with pytest.raises(ValueError, match="findings must be None or a list of strings"):
        _write(mod, findings="not a list")


def test_findings_rejects_non_string_elements(mod):
    with pytest.raises(ValueError, match="findings must be None or a list of strings"):
        _write(mod, findings=["ok", 42])


def test_findings_rejects_nested_list(mod):
    with pytest.raises(ValueError, match="findings must be None or a list of strings"):
        _write(mod, findings=[["nested"]])


def test_bad_findings_writes_nothing(mod):
    """Validation must fire BEFORE the append, so a bad call leaves no row."""
    with pytest.raises(ValueError):
        _write(mod, findings="not a list")
    assert not mod.JSONL_PATH.exists() or mod.JSONL_PATH.read_text().strip() == ""


def test_validate_record_accepts_missing_findings_key(mod):
    """_validate_record is also reached by legacy dicts with no findings key."""
    mod._validate_record(
        {
            "subagent_id": "x",
            "status": "in_progress",
            "step": 1,
            "files_touched": [],
            "next_action": "",
            "notes": "",
        }
    )


# ─── (c) legacy compatibility ────────────────────────────────────────────


def _legacy_row(sid="legacy_arm", step=1, status="in_progress"):
    """A pre-findings row: exactly the fields that existed before this change."""
    return json.dumps(
        {
            "subagent_id": sid,
            "parent_id_or_session": "sess",
            "lane_id": "lane_legacy",
            "step": step,
            "status": status,
            "files_touched": ["a.py"],
            "next_action": "resume here",
            "notes": "lane_legacy",
            "written_at_utc": "2026-08-01T00:00:00+00:00",
            "pid": 1,
            "host": "h",
        },
        sort_keys=True,
    )


def test_legacy_rows_still_load(mod):
    mod.JSONL_PATH.write_text(_legacy_row() + "\n")
    rows = mod.read_checkpoints("legacy_arm")
    assert len(rows) == 1
    assert rows[0].get("findings") is None
    assert rows[0]["next_action"] == "resume here"


def test_every_reader_tolerates_legacy_rows(mod):
    mod.JSONL_PATH.write_text(_legacy_row() + "\n")
    assert mod.latest_checkpoint("legacy_arm") is not None
    assert mod.read_checkpoints_by_parent("sess")
    assert mod.read_checkpoints_by_lane("lane_legacy")
    assert mod.latest_incomplete_checkpoint() is not None
    assert mod.latest_incomplete_for_parent("sess") is not None
    assert mod.latest_incomplete_for_lane("lane_legacy") is not None


def test_legacy_and_new_rows_interleave(mod):
    mod.JSONL_PATH.write_text(_legacy_row() + "\n")
    _write(mod, subagent_id="legacy_arm", step=2, findings=["learned something"])
    rows = mod.read_checkpoints("legacy_arm")
    assert [r.get("findings") for r in rows] == [None, ["learned something"]]


# ─── (d) the knowledge log ───────────────────────────────────────────────


def test_read_findings_one_row_per_finding(mod):
    _write(mod, step=1, findings=["a", "b"])
    _write(mod, step=2, findings=["c"])
    log = mod.read_findings("ddm_test")
    assert [r["finding"] for r in log] == ["a", "b", "c"]
    assert [r["step"] for r in log] == [1, 1, 2]
    assert all(r["subagent_id"] == "ddm_test" for r in log)
    assert all(r["written_at_utc"] for r in log)


def test_read_findings_skips_rows_without_findings(mod):
    _write(mod, step=1)
    _write(mod, step=2, findings=["only this"])
    assert [r["finding"] for r in mod.read_findings("ddm_test")] == ["only this"]


def test_read_findings_whole_fleet(mod):
    _write(mod, subagent_id="arm_a", findings=["from a"])
    _write(mod, subagent_id="arm_b", findings=["from b"])
    log = mod.read_findings()
    assert {r["subagent_id"] for r in log} == {"arm_a", "arm_b"}
    assert len(log) == 2


def test_read_findings_empty_when_none_recorded(mod):
    _write(mod)
    assert mod.read_findings("ddm_test") == []
    assert mod.read_findings() == []


def test_read_findings_on_legacy_only_store_is_empty_not_error(mod):
    """Honest representation: legacy arms recorded no findings, so the log is short."""
    mod.JSONL_PATH.write_text(_legacy_row() + "\n")
    assert mod.read_findings("legacy_arm") == []


# ─── (e) CLI surface ─────────────────────────────────────────────────────


def test_cli_finding_flag_is_repeatable(mod, capsys):
    rc = mod.main(
        [
            "--subagent-id", "cli_arm",
            "--step", "1",
            "--status", "in_progress",
            "--finding", "first, with a comma",
            "--finding", "second",
        ]
    )
    assert rc == 0
    rec = json.loads(capsys.readouterr().out.strip())
    assert rec["findings"] == ["first, with a comma", "second"]


def test_cli_without_finding_writes_none(mod, capsys):
    rc = mod.main(
        ["--subagent-id", "cli_arm", "--step", "1", "--status", "in_progress"]
    )
    assert rc == 0
    assert json.loads(capsys.readouterr().out.strip())["findings"] is None


def test_cli_read_findings_for_one_agent(mod, capsys):
    mod.main(
        ["--subagent-id", "cli_arm", "--step", "1", "--status", "in_progress",
         "--finding", "learned X"]
    )
    capsys.readouterr()
    rc = mod.main(["read", "--findings", "--subagent-id", "cli_arm"])
    assert rc == 0
    assert json.loads(capsys.readouterr().out.strip())["finding"] == "learned X"


def test_cli_read_findings_alone_is_allowed(mod, capsys):
    mod.main(
        ["--subagent-id", "a", "--step", "1", "--status", "in_progress",
         "--finding", "fleet-wide"]
    )
    capsys.readouterr()
    assert mod.main(["read", "--findings"]) == 0
    assert "fleet-wide" in capsys.readouterr().out


def test_cli_read_findings_rc2_when_empty(mod, capsys):
    mod.main(["--subagent-id", "a", "--step", "1", "--status", "in_progress"])
    capsys.readouterr()
    assert mod.main(["read", "--findings", "--subagent-id", "a"]) == 2
    assert "no findings recorded" in capsys.readouterr().err


def test_cli_read_without_any_mode_still_errors(mod):
    """--findings relaxes the query-mode requirement; nothing else does."""
    with pytest.raises(SystemExit):
        mod.main(["read"])


def test_cli_read_findings_refuses_incompatible_query_modes(mod):
    with pytest.raises(SystemExit):
        mod.main(["read", "--findings", "--lane-id", "lane_x"])


def test_cli_read_plain_records_unaffected(mod, capsys):
    """The default read path must be unchanged by this addition."""
    mod.main(
        ["--subagent-id", "a", "--step", "1", "--status", "in_progress",
         "--finding", "f"]
    )
    capsys.readouterr()
    assert mod.main(["read", "--subagent-id", "a"]) == 0
    rec = json.loads(capsys.readouterr().out.strip())
    assert rec["next_action"] == "" and rec["findings"] == ["f"]


# ─── contract wiring ─────────────────────────────────────────────────────


def test_contract_requires_a_finding_every_checkpoint():
    from tac.subagent_contract import (
        CHECKPOINT_FINDINGS,
        CONTRACT_CONSTANT_NAMES,
        KEY_PHRASES,
        standard_contract,
    )

    assert "CHECKPOINT_FINDINGS" in CONTRACT_CONSTANT_NAMES
    assert KEY_PHRASES["CHECKPOINT_FINDINGS"] in CHECKPOINT_FINDINGS
    assert "--finding" in CHECKPOINT_FINDINGS
    assert CHECKPOINT_FINDINGS in standard_contract()
