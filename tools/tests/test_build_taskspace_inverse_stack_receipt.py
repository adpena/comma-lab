from __future__ import annotations

import json

from tools import build_taskspace_inverse_stack_receipt as tool


def test_parser_defaults_to_canonical_output() -> None:
    args = tool.parse_args([])
    assert args.output == tool.DEFAULT_OUTPUT


def test_main_reports_false_authority(monkeypatch, capsys) -> None:
    receipt = {
        "body_sha256": "a" * 64,
        "body": {"verdict": "BLOCKED_RESEARCH_ONLY"},
    }
    writes = []
    monkeypatch.setattr(tool, "build_stack_receipt", lambda **_kwargs: receipt)
    monkeypatch.setattr(tool, "write_once_receipt", lambda *args, **kwargs: writes.append((args, kwargs)))
    assert tool.main([]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["score_claim"] is False
    assert output["promotion_eligible"] is False
    assert output["pointer_moved"] is False
    assert writes
