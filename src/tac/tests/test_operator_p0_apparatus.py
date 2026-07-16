"""Tests for the operator-P0 anti-abandonment apparatus.

Covers tools/operator_p0_digest.py (fcntl-locked latest-wins ledger + the
SessionStart/compact digest) and tools/operator_p0_stop_hook.py (the Stop-hook
demand-update nag) — pure logic in tmpdirs + fail-open integration smokes on
the real repo, mirroring test_triality_drift_detector.py conventions.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import subprocess
import sys

_REPO = pathlib.Path(__file__).resolve().parents[3]
_DIGEST = _REPO / "tools" / "operator_p0_digest.py"
_HOOK = _REPO / "tools" / "operator_p0_stop_hook.py"


def _load(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


digest = _load(_DIGEST, "operator_p0_digest_test")
hook = _load(_HOOK, "operator_p0_stop_hook_test")


def _row(pid="p0_test", status="open", **kw):
    base = {
        "p0_id": pid,
        "designated_date": "2026-07-15",
        "verbatim_ask": "operator: test ask",
        "status": status,
        "evidence": "test evidence",
        "next_action": "test next",
        "source": "test",
    }
    base.update(kw)
    return base


# ------------------------------ ledger library ------------------------------
def test_append_and_read_latest_wins(tmp_path):
    root = str(tmp_path)
    digest.append_row(root, _row(status="open"))
    digest.append_row(root, _row(status="complete", evidence="done artifact"))
    latest = digest.read_ledger(root)
    assert latest["p0_test"]["status"] == "complete"
    assert len(digest.read_rows(root)) == 2  # append-only history preserved


def test_append_rejects_missing_required_field(tmp_path):
    root = str(tmp_path)
    bad = _row()
    bad["verbatim_ask"] = ""
    try:
        digest.append_row(root, bad)
        raise AssertionError("expected ValueError")
    except ValueError as e:
        assert "verbatim_ask" in str(e)


def test_append_rejects_bad_status(tmp_path):
    root = str(tmp_path)
    try:
        digest.append_row(root, _row(status="abandoned"))
        raise AssertionError("expected ValueError")
    except ValueError as e:
        assert "status" in str(e)


def test_max_written_utc_moves_on_append(tmp_path):
    root = str(tmp_path)
    assert digest.max_written_utc(root) == ""
    digest.append_row(root, _row())
    assert digest.max_written_utc(root) > ""


def test_open_rows_orders_open_before_in_progress(tmp_path):
    root = str(tmp_path)
    digest.append_row(root, _row(pid="p0_b", status="in_progress"))
    digest.append_row(root, _row(pid="p0_a", status="open"))
    digest.append_row(root, _row(pid="p0_c", status="complete"))
    rows = digest.open_rows(root)
    assert [r["p0_id"] for r in rows] == ["p0_a", "p0_b"]  # complete excluded


def test_format_digest_names_ids_and_next_action(tmp_path):
    root = str(tmp_path)
    digest.append_row(root, _row(next_action="do the thing"))
    text = digest.format_digest(digest.open_rows(root))
    assert "p0_test" in text and "do the thing" in text and "OPERATOR-P0" in text


def test_read_rows_skips_malformed_lines(tmp_path):
    root = str(tmp_path)
    digest.append_row(root, _row())
    with open(digest.ledger_path(root), "a") as fh:
        fh.write("not json\n{\"p0_id\": 3}\n")
    assert len(digest.read_rows(root)) >= 1  # never raises


def test_update_cli_roundtrip(tmp_path):
    root = str(tmp_path)
    rc = digest.main([
        "--root", root, "--update", "p0_cli", "--status", "open",
        "--verbatim-ask", "op ask", "--evidence", "ev", "--next-action", "nx",
        "--source", "test", "--designated-date", "2026-07-15",
    ])
    assert rc == 0
    # inherit-prior semantics: a later update keeps unspecified fields
    rc = digest.main(["--root", root, "--update", "p0_cli", "--status", "complete"])
    assert rc == 0
    latest = digest.read_ledger(root)["p0_cli"]
    assert latest["status"] == "complete" and latest["verbatim_ask"] == "op ask"


def test_session_start_mode_exits_zero_even_on_bad_root():
    p = subprocess.run(
        [sys.executable, str(_DIGEST), "--session-start", "--root", "/nonexistent-xyz"],
        capture_output=True, text=True, timeout=30,
    )
    assert p.returncode == 0


def test_session_start_on_real_repo_prints_open_p0s():
    p = subprocess.run(
        [sys.executable, str(_DIGEST), "--session-start", "--root", str(_REPO)],
        capture_output=True, text=True, timeout=30,
    )
    assert p.returncode == 0
    # the seeded recovery ledger has open rows; the digest must surface them
    assert "OPERATOR-P0 LEDGER" in p.stdout


# ------------------------------ stop-hook logic ------------------------------
def test_touched_p0s_by_p0_id_in_subject():
    rows = [_row(pid="p0_366_joint_pose", status="open")]
    hit = hook.touched_p0s(rows, ["progress on p0_366_joint_pose run prep"], [])
    assert hit == ["p0_366_joint_pose"]


def test_touched_p0s_by_task_number():
    rows = [_row(pid="p0_x", status="open", task_ids=["366"])]
    assert hook.touched_p0s(rows, ["launch prep for #366 finishing run"], []) == ["p0_x"]
    assert hook.touched_p0s(rows, ["mentions #3660 which is different"], []) == []


def test_touched_p0s_by_watch_path():
    rows = [_row(pid="p0_w", status="in_progress",
                 watch_paths=["src/tac/boundary_math/phase_residual_carrier.py"])]
    files = ["src/tac/boundary_math/phase_residual_carrier.py"]
    assert hook.touched_p0s(rows, ["chore"], files) == ["p0_w"]


def test_touched_p0s_ignores_complete_rows():
    rows = [_row(pid="p0_done", status="complete", task_ids=["366"])]
    assert hook.touched_p0s(rows, ["#366 work"], []) == []


def test_opt_out_token():
    assert hook.is_opted_out(["mechanical chore [p0-ledger-ok]"])
    assert not hook.is_opted_out(["ordinary commit"])


def _user_line(text):
    return json.dumps({"type": "user",
                       "message": {"role": "user",
                                   "content": [{"type": "text", "text": text}]}})


def test_new_designation_fires_on_directive_p0():
    lines = [_user_line("Pursue the flicker stack as p0 now, no more deferring")]
    hits = hook.new_p0_designations(lines)
    assert len(hits) == 1 and "p0" in hits[0].lower()


def test_new_designation_silent_on_plain_mention():
    # word "p0" without a directive verb on the same line stays silent
    lines = [_user_line("the p0 column in that table looks fine")]
    assert hook.new_p0_designations(lines) == []


def test_new_designation_silent_on_assistant_and_tool_result():
    a = json.dumps({"type": "assistant",
                    "message": {"role": "assistant",
                                "content": [{"type": "text",
                                             "text": "pursue this as p0 now"}]}})
    t = json.dumps({"type": "user",
                    "message": {"role": "user",
                                "content": [{"type": "tool_result",
                                             "content": "pursue as p0 now"}]}})
    assert hook.new_p0_designations([a, t]) == []


def test_new_designation_silent_on_apparatus_self_reference():
    lines = [_user_line("update the operator_p0 ledger and pursue the digest as p0 now")]
    assert hook.new_p0_designations(lines) == []


def test_new_designation_word_boundary():
    # "p0" inside identifiers (fp0, p05) must not fire
    lines = [_user_line("must fix the fp0 register and p05 build now")]
    assert hook.new_p0_designations(lines) == []


def test_extract_text_variants():
    assert hook.extract_text("plain") == "plain"
    assert hook.extract_text({"content": "s"}) == "s"
    assert hook.extract_text({"content": [{"type": "text", "text": "a"}]}) == "a"
    assert hook.extract_text(None) == ""


# ------------------------------ fail-open smokes ------------------------------
def test_hook_exits_zero_on_empty_stdin():
    p = subprocess.run([sys.executable, str(_HOOK)], input="", capture_output=True,
                       text=True, timeout=30, cwd=str(_REPO))
    assert p.returncode == 0


def test_hook_exits_zero_on_garbage_stdin():
    p = subprocess.run([sys.executable, str(_HOOK)], input="{not json",
                       capture_output=True, text=True, timeout=30, cwd=str(_REPO))
    assert p.returncode == 0


def test_hook_stop_hook_active_allows():
    p = subprocess.run(
        [sys.executable, str(_HOOK)],
        input=json.dumps({"stop_hook_active": True, "cwd": str(_REPO)}),
        capture_output=True, text=True, timeout=30, cwd=str(_REPO),
    )
    assert p.returncode == 0
    assert '"decision": "block"' not in p.stdout
