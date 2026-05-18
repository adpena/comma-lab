from __future__ import annotations

import datetime as dt
import json
import multiprocessing as mp
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tac.codex_to_claude_inbox import (
    INBOX_SCHEMA_VERSION,
    STATUS_ANSWERED,
    STATUS_OPEN,
    STATUS_OPERATOR_DEFAULT_INVOKED,
    STATUS_WITHDRAWN,
    VALID_EVENT_TYPES,
    VALID_STATUSES,
    InboxRowCorruptError,
    InboxRowValidationError,
    append_inbox_ack,
    append_inbox_answer,
    append_inbox_operator_default_invoked,
    append_inbox_question,
    append_inbox_relay,
    append_inbox_withdraw,
    inbox_summary,
    latest_status_by_event_id,
    load_inbox_strict,
    query_by_event_id,
    query_open_questions_for_claude,
    query_unread_relays,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
CLI = REPO_ROOT / "tools" / "codex_to_claude_inbox.py"


@pytest.fixture
def tmp_inbox(tmp_path: Path) -> tuple[Path, Path]:
    path = tmp_path / "codex_to_claude_inbox.jsonl"
    lock = tmp_path / "codex_to_claude_inbox.jsonl.lock"
    return path, lock


def _past_deadline() -> str:
    return (dt.datetime.now(dt.UTC) - dt.timedelta(seconds=2)).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _future_deadline() -> str:
    return (dt.datetime.now(dt.UTC) + dt.timedelta(hours=4)).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def test_schema_invariants() -> None:
    assert INBOX_SCHEMA_VERSION == "codex_to_claude_inbox_v1_20260518"
    assert {
        "question",
        "relay",
        "answer",
        "ack",
        "operator_default_invoked",
        "withdraw",
    } == VALID_EVENT_TYPES
    assert {
        "open",
        "answered",
        "operator_default_invoked",
        "stale",
        "withdrawn",
    } == VALID_STATUSES


def test_append_question_happy_path(tmp_inbox: tuple[Path, Path]) -> None:
    path, lock = tmp_inbox
    row = append_inbox_question(
        question_text="Should this task fail closed?",
        blocking_task_id="task_1",
        context_pointers=("a.md", "b.md"),
        suggested_options=("fail-closed", "defer"),
        codex_default_if_no_response="fail-closed",
        response_deadline_utc=_future_deadline(),
        path=path,
        lock_path=lock,
        event_id="question_fixed",
    )
    assert row["event_id"] == "question_fixed"
    assert row["status"] == STATUS_OPEN
    assert load_inbox_strict(path)[0]["question_text"] == "Should this task fail closed?"


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"question_text": ""}, "question_text"),
        ({"event_id": ""}, "event_id"),
        ({"event_id": "bad\nid"}, "event_id"),
        ({"response_deadline_utc": "not-a-date"}, "timestamp"),
        ({"agent": ""}, "agent"),
        ({"codex_default_if_no_response": ""}, "codex_default"),
        ({"schema_version": "bad"}, "reserved"),
    ],
)
def test_append_question_rejects_invalid_inputs(
    tmp_inbox: tuple[Path, Path],
    kwargs: dict[str, str],
    match: str,
) -> None:
    path, lock = tmp_inbox
    base = {"question_text": "Q?", "path": path, "lock_path": lock}
    base.update(kwargs)
    with pytest.raises((InboxRowValidationError, TypeError), match=match):
        append_inbox_question(**base)


def test_relay_is_not_open_question(tmp_inbox: tuple[Path, Path]) -> None:
    path, lock = tmp_inbox
    relay = append_inbox_relay(
        relay_text="Leaderboard moved; intake started.",
        context_pointers=("reports/latest.md",),
        path=path,
        lock_path=lock,
    )
    assert relay["status"] == STATUS_ANSWERED
    assert query_open_questions_for_claude(path=path) == []


def test_answer_validates_open_question(tmp_inbox: tuple[Path, Path]) -> None:
    path, lock = tmp_inbox
    q = append_inbox_question(question_text="Q?", path=path, lock_path=lock)
    answer = append_inbox_answer(
        response_to_event_id=q["event_id"],
        answer_text="Prefer fail-closed.",
        answer_memo_path=".omx/research/claude_response.md",
        path=path,
        lock_path=lock,
    )
    assert answer["status"] == STATUS_ANSWERED
    assert latest_status_by_event_id(path=path)[q["event_id"]] == STATUS_ANSWERED
    assert len(query_by_event_id(q["event_id"], path=path)) == 2


def test_answer_rejects_missing_question(tmp_inbox: tuple[Path, Path]) -> None:
    path, lock = tmp_inbox
    with pytest.raises(InboxRowValidationError, match="not found"):
        append_inbox_answer(
            response_to_event_id="missing",
            answer_text="No.",
            path=path,
            lock_path=lock,
        )


def test_answer_rejects_closed_question(tmp_inbox: tuple[Path, Path]) -> None:
    path, lock = tmp_inbox
    q = append_inbox_question(question_text="Q?", path=path, lock_path=lock)
    append_inbox_answer(
        response_to_event_id=q["event_id"],
        answer_text="Done.",
        path=path,
        lock_path=lock,
    )
    with pytest.raises(InboxRowValidationError, match="not open"):
        append_inbox_answer(
            response_to_event_id=q["event_id"],
            answer_text="Second.",
            path=path,
            lock_path=lock,
        )


def test_operator_default_requires_expired_deadline(tmp_inbox: tuple[Path, Path]) -> None:
    path, lock = tmp_inbox
    q = append_inbox_question(
        question_text="Q?",
        codex_default_if_no_response="fail-closed",
        response_deadline_utc=_future_deadline(),
        path=path,
        lock_path=lock,
    )
    with pytest.raises(InboxRowValidationError, match="before response deadline"):
        append_inbox_operator_default_invoked(
            response_to_event_id=q["event_id"],
            default_used="fail-closed",
            path=path,
            lock_path=lock,
        )


def test_operator_default_invoked_after_deadline(tmp_inbox: tuple[Path, Path]) -> None:
    path, lock = tmp_inbox
    q = append_inbox_question(
        question_text="Q?",
        codex_default_if_no_response="fail-closed",
        response_deadline_utc=_past_deadline(),
        path=path,
        lock_path=lock,
    )
    row = append_inbox_operator_default_invoked(
        response_to_event_id=q["event_id"],
        default_used="fail-closed",
        path=path,
        lock_path=lock,
    )
    assert row["status"] == STATUS_OPERATOR_DEFAULT_INVOKED
    assert latest_status_by_event_id(path=path)[q["event_id"]] == STATUS_OPERATOR_DEFAULT_INVOKED


def test_operator_default_rejects_question_without_deadline(tmp_inbox: tuple[Path, Path]) -> None:
    path, lock = tmp_inbox
    q = append_inbox_question(
        question_text="Q?",
        codex_default_if_no_response="fail-closed",
        path=path,
        lock_path=lock,
    )
    with pytest.raises(InboxRowValidationError, match="without response_deadline_utc"):
        append_inbox_operator_default_invoked(
            response_to_event_id=q["event_id"],
            default_used="fail-closed",
            path=path,
            lock_path=lock,
        )


def test_operator_default_requires_declared_default(tmp_inbox: tuple[Path, Path]) -> None:
    path, lock = tmp_inbox
    q = append_inbox_question(
        question_text="Q?",
        response_deadline_utc=_past_deadline(),
        path=path,
        lock_path=lock,
    )
    with pytest.raises(InboxRowValidationError, match="codex_default_if_no_response"):
        append_inbox_operator_default_invoked(
            response_to_event_id=q["event_id"],
            default_used="fail-closed",
            path=path,
            lock_path=lock,
        )


def test_operator_default_must_match_declared_default(tmp_inbox: tuple[Path, Path]) -> None:
    path, lock = tmp_inbox
    q = append_inbox_question(
        question_text="Q?",
        codex_default_if_no_response="fail-closed",
        response_deadline_utc=_past_deadline(),
        path=path,
        lock_path=lock,
    )
    with pytest.raises(InboxRowValidationError, match="default_used"):
        append_inbox_operator_default_invoked(
            response_to_event_id=q["event_id"],
            default_used="defer",
            path=path,
            lock_path=lock,
        )


def test_ack_preserves_latest_status(tmp_inbox: tuple[Path, Path]) -> None:
    path, lock = tmp_inbox
    q = append_inbox_question(question_text="Q?", path=path, lock_path=lock)
    append_inbox_answer(response_to_event_id=q["event_id"], answer_text="A.", path=path, lock_path=lock)
    ack = append_inbox_ack(response_to_event_id=q["event_id"], ack_text="Read.", path=path, lock_path=lock)
    assert ack["status"] == STATUS_ANSWERED
    assert latest_status_by_event_id(path=path)[q["event_id"]] == STATUS_ANSWERED


def test_withdraw_open_question(tmp_inbox: tuple[Path, Path]) -> None:
    path, lock = tmp_inbox
    q = append_inbox_question(question_text="Q?", path=path, lock_path=lock)
    row = append_inbox_withdraw(
        response_to_event_id=q["event_id"],
        reason="Answered by adjacent memo.",
        path=path,
        lock_path=lock,
    )
    assert row["status"] == STATUS_WITHDRAWN
    assert query_open_questions_for_claude(path=path) == []


def test_withdraw_rejects_closed_question(tmp_inbox: tuple[Path, Path]) -> None:
    path, lock = tmp_inbox
    q = append_inbox_question(question_text="Q?", path=path, lock_path=lock)
    append_inbox_answer(response_to_event_id=q["event_id"], answer_text="A.", path=path, lock_path=lock)
    with pytest.raises(InboxRowValidationError, match="not open"):
        append_inbox_withdraw(
            response_to_event_id=q["event_id"],
            reason="Too late.",
            path=path,
            lock_path=lock,
        )


def test_ack_rejects_missing_event(tmp_inbox: tuple[Path, Path]) -> None:
    path, lock = tmp_inbox
    with pytest.raises(InboxRowValidationError, match="event not found"):
        append_inbox_ack(
            response_to_event_id="missing",
            ack_text="Read.",
            path=path,
            lock_path=lock,
        )


def test_query_open_questions_for_claude_filters_agent(tmp_inbox: tuple[Path, Path]) -> None:
    path, lock = tmp_inbox
    append_inbox_question(question_text="Codex Q?", agent="codex", path=path, lock_path=lock)
    append_inbox_question(question_text="Other Q?", agent="operator", path=path, lock_path=lock)
    open_rows = query_open_questions_for_claude(path=path)
    assert len(open_rows) == 1
    assert open_rows[0]["question_text"] == "Codex Q?"


def test_query_unread_relays_respects_ack(tmp_inbox: tuple[Path, Path]) -> None:
    path, lock = tmp_inbox
    relay = append_inbox_relay(relay_text="Info.", path=path, lock_path=lock)
    assert len(query_unread_relays(since_utc="2026-01-01T00:00:00Z", path=path)) == 1
    append_inbox_ack(response_to_event_id=relay["event_id"], ack_text="Read.", path=path, lock_path=lock)
    assert query_unread_relays(since_utc="2026-01-01T00:00:00Z", path=path) == []


def test_load_inbox_strict_quarantines_corrupt_json(tmp_path: Path) -> None:
    path = tmp_path / "codex_to_claude_inbox.jsonl"
    path.write_text("{not-json}\n", encoding="utf-8")
    with pytest.raises(InboxRowCorruptError):
        load_inbox_strict(path)
    assert not path.exists()
    assert list(tmp_path.glob("codex_to_claude_inbox.jsonl.corrupt.*"))


def test_jsonl_byte_stable_format(tmp_inbox: tuple[Path, Path]) -> None:
    path, lock = tmp_inbox
    append_inbox_question(question_text="Q?", path=path, lock_path=lock, event_id="question_fixed")
    line = path.read_text(encoding="utf-8")
    parsed = json.loads(line)
    assert json.dumps(parsed, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n" == line


def test_no_tmp_files_leak_on_success(tmp_inbox: tuple[Path, Path]) -> None:
    path, lock = tmp_inbox
    append_inbox_question(question_text="Q?", path=path, lock_path=lock)
    assert not list(path.parent.glob("*.tmp.*"))


def test_custom_path_without_lock_path_uses_adjacent_lock(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "custom_inbox.jsonl"
    question = append_inbox_question(question_text="Q?", path=path)
    append_inbox_answer(response_to_event_id=question["event_id"], answer_text="A.", path=path)
    assert path.with_suffix(path.suffix + ".lock").exists()
    assert latest_status_by_event_id(path=path)[question["event_id"]] == STATUS_ANSWERED


def _worker_append(path_text: str, lock_text: str, worker_idx: int) -> None:
    path = Path(path_text)
    lock = Path(lock_text)
    for i in range(5):
        append_inbox_question(
            question_text=f"worker {worker_idx} question {i}",
            path=path,
            lock_path=lock,
        )


def _worker_answer(path_text: str, lock_text: str, event_id: str, worker_idx: int, queue: mp.Queue) -> None:
    path = Path(path_text)
    lock = Path(lock_text)
    try:
        row = append_inbox_answer(
            response_to_event_id=event_id,
            answer_text=f"worker {worker_idx} answer",
            path=path,
            lock_path=lock,
        )
    except Exception as exc:  # pragma: no cover - exercised in subprocess.
        queue.put(("err", type(exc).__name__, str(exc)))
    else:
        queue.put(("ok", row["event_id"], row["status"]))


def _worker_ack_or_answer(path_text: str, lock_text: str, event_id: str, action: str, queue: mp.Queue) -> None:
    path = Path(path_text)
    lock = Path(lock_text)
    try:
        if action == "ack":
            row = append_inbox_ack(
                response_to_event_id=event_id,
                ack_text="Read.",
                path=path,
                lock_path=lock,
            )
        else:
            row = append_inbox_answer(
                response_to_event_id=event_id,
                answer_text="Resolved.",
                path=path,
                lock_path=lock,
            )
    except Exception as exc:  # pragma: no cover - exercised in subprocess.
        queue.put(("err", action, type(exc).__name__, str(exc)))
    else:
        queue.put(("ok", action, row["event_id"], row["status"]))


def test_concurrent_append_stress(tmp_inbox: tuple[Path, Path]) -> None:
    path, lock = tmp_inbox
    procs = [
        mp.Process(target=_worker_append, args=(str(path), str(lock), idx))
        for idx in range(4)
    ]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(timeout=10)
    assert all(proc.exitcode == 0 for proc in procs)
    rows = load_inbox_strict(path)
    assert len(rows) == 20
    assert len({row["event_id"] for row in rows}) == 20


def test_concurrent_answer_resolution_has_single_winner(tmp_inbox: tuple[Path, Path]) -> None:
    path, lock = tmp_inbox
    question = append_inbox_question(question_text="Race?", path=path, lock_path=lock)
    queue: mp.Queue = mp.Queue()
    procs = [
        mp.Process(target=_worker_answer, args=(str(path), str(lock), question["event_id"], idx, queue))
        for idx in range(2)
    ]
    for proc in procs:
        proc.start()
    results = [queue.get(timeout=10) for _ in procs]
    for proc in procs:
        proc.join(timeout=10)

    assert all(proc.exitcode == 0 for proc in procs)
    assert sum(result[0] == "ok" for result in results) == 1
    assert sum(result[0] == "err" and result[1] == "InboxRowValidationError" for result in results) == 1
    assert len(query_by_event_id(question["event_id"], path=path)) == 2
    assert latest_status_by_event_id(path=path)[question["event_id"]] == STATUS_ANSWERED


def test_ack_does_not_reopen_resolved_question_under_race(tmp_inbox: tuple[Path, Path]) -> None:
    path, lock = tmp_inbox
    question = append_inbox_question(question_text="Ack race?", path=path, lock_path=lock)
    queue: mp.Queue = mp.Queue()
    procs = [
        mp.Process(target=_worker_ack_or_answer, args=(str(path), str(lock), question["event_id"], action, queue))
        for action in ("ack", "answer")
    ]
    for proc in procs:
        proc.start()
    results = [queue.get(timeout=10) for _ in procs]
    for proc in procs:
        proc.join(timeout=10)

    assert all(proc.exitcode == 0 for proc in procs)
    assert any(result[0] == "ok" and result[1] == "answer" for result in results)
    assert latest_status_by_event_id(path=path)[question["event_id"]] == STATUS_ANSWERED


def test_inbox_summary_counts_expired_open_questions(tmp_inbox: tuple[Path, Path]) -> None:
    path, lock = tmp_inbox
    q = append_inbox_question(
        question_text="Q?",
        response_deadline_utc=_past_deadline(),
        path=path,
        lock_path=lock,
    )
    summary = inbox_summary(path=path)
    assert summary["open_questions_count"] == 1
    assert summary["expired_open_questions_count"] == 1
    assert summary["expired_open_question_event_ids"] == [q["event_id"]]


def test_cli_help_runs() -> None:
    proc = subprocess.run(
        [sys.executable, str(CLI), "--help"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "ask" in proc.stdout


def test_cli_ask_poll_answer_summary_lifecycle(tmp_path: Path) -> None:
    path = tmp_path / "codex_to_claude_inbox.jsonl"
    lock = tmp_path / "codex_to_claude_inbox.jsonl.lock"
    env = {
        **os.environ,
        "PACT_CODEX_INBOX_PATH": str(path),
        "PACT_CODEX_INBOX_LOCK_PATH": str(lock),
    }
    ask = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "ask",
            "--question",
            "Should Codex fail closed?",
            "--suggested-options",
            "fail-closed|defer",
            "--codex-default-if-no-response",
            "fail-closed",
            "--response-deadline-utc",
            _future_deadline(),
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    question = json.loads(ask.stdout)
    assert question["status"] == STATUS_OPEN

    poll = subprocess.run(
        [sys.executable, str(CLI), "poll-for-claude"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    assert question["event_id"] in poll.stdout

    subprocess.run(
        [
            sys.executable,
            str(CLI),
            "answer",
            "--response-to-event-id",
            question["event_id"],
            "--answer",
            "Yes, fail closed.",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    summary = subprocess.run(
        [sys.executable, str(CLI), "summary", "--format", "json"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(summary.stdout)
    assert payload["open_questions_count"] == 0
    assert payload["latest_status_by_event_id"][question["event_id"]] == STATUS_ANSWERED


def test_cli_compact_json_is_single_line(tmp_path: Path) -> None:
    path = tmp_path / "codex_to_claude_inbox.jsonl"
    lock = tmp_path / "codex_to_claude_inbox.jsonl.lock"
    env = {
        **os.environ,
        "PACT_CODEX_INBOX_PATH": str(path),
        "PACT_CODEX_INBOX_LOCK_PATH": str(lock),
    }
    proc = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "--compact-json",
            "ask",
            "--question",
            "Compact?",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    assert proc.stdout.count("\n") == 1
    assert json.loads(proc.stdout)["question_text"] == "Compact?"


def test_preflight_gate_flags_expired_open_question(tmp_path: Path) -> None:
    from tac.preflight import check_codex_inbox_open_questions_have_response_or_default_within_deadline

    path = tmp_path / ".omx" / "state" / "codex_to_claude_inbox.jsonl"
    lock = path.with_suffix(path.suffix + ".lock")
    append_inbox_question(
        question_text="Expired?",
        response_deadline_utc=_past_deadline(),
        path=path,
        lock_path=lock,
        event_id="question_expired",
    )
    violations = check_codex_inbox_open_questions_have_response_or_default_within_deadline(
        repo_root=tmp_path,
        strict=False,
        verbose=False,
    )
    assert violations
    with pytest.raises(Exception, match="past deadline"):
        check_codex_inbox_open_questions_have_response_or_default_within_deadline(
            repo_root=tmp_path,
            strict=True,
            verbose=False,
        )


def test_preflight_gate_ignores_answered_expired_question(tmp_path: Path) -> None:
    from tac.preflight import check_codex_inbox_open_questions_have_response_or_default_within_deadline

    path = tmp_path / ".omx" / "state" / "codex_to_claude_inbox.jsonl"
    lock = path.with_suffix(path.suffix + ".lock")
    q = append_inbox_question(
        question_text="Expired but answered?",
        response_deadline_utc=_past_deadline(),
        path=path,
        lock_path=lock,
    )
    append_inbox_answer(
        response_to_event_id=q["event_id"],
        answer_text="Resolved.",
        path=path,
        lock_path=lock,
    )
    assert (
        check_codex_inbox_open_questions_have_response_or_default_within_deadline(
            repo_root=tmp_path,
            strict=True,
            verbose=False,
        )
        == []
    )


def test_preflight_registers_inbox_bare_write_contract() -> None:
    source = (REPO_ROOT / "src/tac/preflight.py").read_text(encoding="utf-8")
    assert "INBOX_PATH" in source
    assert "_inbox_lock" in source
    assert "append_inbox_question" in source
    assert "src/tac/codex_to_claude_inbox.py" in source
