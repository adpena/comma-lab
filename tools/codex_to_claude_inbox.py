#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Operator CLI for the canonical Codex to Claude inbox."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tac.codex_to_claude_inbox import (  # noqa: E402
    append_inbox_ack,
    append_inbox_answer,
    append_inbox_operator_default_invoked,
    append_inbox_question,
    append_inbox_relay,
    append_inbox_withdraw,
    inbox_summary,
    query_by_event_id,
    query_open_questions_for_claude,
)


def _split_csv(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _split_options(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(part.strip() for part in value.split("|") if part.strip())


_COMPACT_JSON = False


def _print_json(payload: object) -> None:
    if _COMPACT_JSON:
        print(json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")))
    else:
        print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))


def _cli_path() -> Path | None:
    value = os.environ.get("PACT_CODEX_INBOX_PATH")
    return Path(value) if value else None


def _cli_lock_path() -> Path | None:
    value = os.environ.get("PACT_CODEX_INBOX_LOCK_PATH")
    return Path(value) if value else None


def _print_summary_text(payload: dict[str, object]) -> None:
    print("Codex to Claude inbox")
    print(f"  rows:                    {payload['row_count']}")
    print(f"  events:                  {payload['event_count']}")
    print(f"  open_questions:          {payload['open_questions_count']}")
    print(f"  expired_open_questions:  {payload['expired_open_questions_count']}")
    print(f"  oldest_open_age_hours:   {payload['open_questions_oldest_age_hours']}")
    print(f"  relays:                  {payload['relays_count']}")
    expired = payload.get("expired_open_question_event_ids") or []
    if expired:
        print("  expired_event_ids:")
        for event_id in expired:
            print(f"    - {event_id}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--compact-json",
        action="store_true",
        help="Emit minified JSON for token-efficient machine consumption.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    ask = sub.add_parser("ask", help="Append an open Codex question for Claude")
    ask.add_argument("--blocking-task-id")
    ask.add_argument("--question", required=True)
    ask.add_argument("--context-pointers", default="")
    ask.add_argument("--suggested-options", default="")
    ask.add_argument("--codex-default-if-no-response")
    ask.add_argument("--response-deadline-utc")
    ask.add_argument("--agent", default="codex")
    ask.add_argument("--subagent-id")
    ask.add_argument("--session-id")

    relay = sub.add_parser("relay", help="Append an informational relay")
    relay.add_argument("--relay", required=True)
    relay.add_argument("--context-pointers", default="")
    relay.add_argument("--agent", default="codex")
    relay.add_argument("--subagent-id")
    relay.add_argument("--session-id")

    poll = sub.add_parser("poll-for-claude", help="Print open Codex questions for Claude")
    poll.add_argument("--format", choices=("json", "text"), default="json")

    answer = sub.add_parser("answer", help="Append a Claude answer to an open question")
    answer.add_argument("--response-to-event-id", required=True)
    answer.add_argument("--answer", required=True)
    answer.add_argument("--answer-memo-path")
    answer.add_argument("--agent", default="claude")
    answer.add_argument("--subagent-id")
    answer.add_argument("--session-id")

    default = sub.add_parser(
        "operator-default-invoked",
        help="Append an operator-default invocation for an expired open question",
    )
    default.add_argument("--response-to-event-id", required=True)
    default.add_argument("--default-used", required=True)
    default.add_argument("--agent", default="codex")
    default.add_argument("--subagent-id")
    default.add_argument("--session-id")

    ack = sub.add_parser("ack", help="Append an acknowledgement row")
    ack.add_argument("--response-to-event-id", required=True)
    ack.add_argument("--ack", required=True)
    ack.add_argument("--agent", default="codex")
    ack.add_argument("--subagent-id")
    ack.add_argument("--session-id")

    withdraw = sub.add_parser("withdraw", help="Withdraw an open question")
    withdraw.add_argument("--event-id", required=True)
    withdraw.add_argument("--reason", required=True)
    withdraw.add_argument("--agent", default="codex")
    withdraw.add_argument("--subagent-id")
    withdraw.add_argument("--session-id")

    by_id = sub.add_parser("show", help="Show all rows for an event id")
    by_id.add_argument("--event-id", required=True)

    summary = sub.add_parser("summary", help="Print inbox summary")
    summary.add_argument("--format", choices=("text", "json"), default="text")
    summary.add_argument("--since-utc")

    return parser


def main(argv: list[str] | None = None) -> int:
    global _COMPACT_JSON
    parser = build_parser()
    args = parser.parse_args(argv)
    _COMPACT_JSON = bool(args.compact_json)
    try:
        if args.cmd == "ask":
            _print_json(
                append_inbox_question(
                    blocking_task_id=args.blocking_task_id,
                    question_text=args.question,
                    context_pointers=_split_csv(args.context_pointers),
                    suggested_options=_split_options(args.suggested_options),
                    codex_default_if_no_response=args.codex_default_if_no_response,
                    response_deadline_utc=args.response_deadline_utc,
                    agent=args.agent,
                    subagent_id=args.subagent_id,
                    session_id=args.session_id,
                    path=_cli_path(),
                    lock_path=_cli_lock_path(),
                )
            )
            return 0
        if args.cmd == "relay":
            _print_json(
                append_inbox_relay(
                    relay_text=args.relay,
                    context_pointers=_split_csv(args.context_pointers),
                    agent=args.agent,
                    subagent_id=args.subagent_id,
                    session_id=args.session_id,
                    path=_cli_path(),
                    lock_path=_cli_lock_path(),
                )
            )
            return 0
        if args.cmd == "poll-for-claude":
            rows = query_open_questions_for_claude(path=_cli_path())
            if args.format == "json":
                _print_json(rows)
            else:
                if not rows:
                    print("No open Codex questions for Claude.")
                for row in rows:
                    print(f"{row['event_id']}: {row['question_text']}")
            return 0
        if args.cmd == "answer":
            _print_json(
                append_inbox_answer(
                    response_to_event_id=args.response_to_event_id,
                    answer_text=args.answer,
                    answer_memo_path=args.answer_memo_path,
                    agent=args.agent,
                    subagent_id=args.subagent_id,
                    session_id=args.session_id,
                    path=_cli_path(),
                    lock_path=_cli_lock_path(),
                )
            )
            return 0
        if args.cmd == "operator-default-invoked":
            _print_json(
                append_inbox_operator_default_invoked(
                    response_to_event_id=args.response_to_event_id,
                    default_used=args.default_used,
                    agent=args.agent,
                    subagent_id=args.subagent_id,
                    session_id=args.session_id,
                    path=_cli_path(),
                    lock_path=_cli_lock_path(),
                )
            )
            return 0
        if args.cmd == "ack":
            _print_json(
                append_inbox_ack(
                    response_to_event_id=args.response_to_event_id,
                    ack_text=args.ack,
                    agent=args.agent,
                    subagent_id=args.subagent_id,
                    session_id=args.session_id,
                    path=_cli_path(),
                    lock_path=_cli_lock_path(),
                )
            )
            return 0
        if args.cmd == "withdraw":
            _print_json(
                append_inbox_withdraw(
                    response_to_event_id=args.event_id,
                    reason=args.reason,
                    agent=args.agent,
                    subagent_id=args.subagent_id,
                    session_id=args.session_id,
                    path=_cli_path(),
                    lock_path=_cli_lock_path(),
                )
            )
            return 0
        if args.cmd == "show":
            _print_json(query_by_event_id(args.event_id, path=_cli_path()))
            return 0
        if args.cmd == "summary":
            payload = inbox_summary(path=_cli_path(), since_utc=args.since_utc)
            if args.format == "json":
                _print_json(payload)
            else:
                _print_summary_text(payload)
            return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    parser.error(f"unknown command: {args.cmd}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
