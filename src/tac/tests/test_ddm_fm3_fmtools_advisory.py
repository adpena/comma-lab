# SPDX-License-Identifier: MIT
"""Tests for the ``ddm_fm3`` fmtools ADVISORY bridge and its Catalog #344 column.

The bridge's whole contract is what it does when things go WRONG, because an
advisory lane that raises, blocks, or fabricates a label is worse than no lane
at all. So most of these tests drive failure paths: no venv, non-zero exit,
timeout, garbage on stdout, per-row errors.

The one invariant that must never break: **a missing or failed advisory lane is
"no advice", never agreement and never a negative.** ``ran=False`` is how the
caller knows to say "fmtools confirmation owed" instead of implying the model
looked and approved.

No test here requires the on-device model. The subprocess is replaced with a
stub, so these run on any machine — which is the point: the lane's fail-open
behaviour must be verifiable where the lane itself cannot run.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.fmtools_advisory import (  # noqa: E402
    AdvisoryVerdict,
    classify_texts,
    fmtools_python,
    log_disagreements,
    unavailable_label,
)

LABELS = ["states_measured_finding", "review_or_process"]


class _Completed:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _stub_run(monkeypatch, *, result=None, raises=None, capture=None):
    """Replace the subprocess call and the interpreter probe."""

    def fake(argv, **kwargs):
        if capture is not None:
            capture["argv"] = argv
            capture["input"] = kwargs.get("input", "")
            capture["timeout"] = kwargs.get("timeout")
        if raises is not None:
            raise raises
        return result

    monkeypatch.setattr(subprocess, "run", fake)
    monkeypatch.setattr("tools.fmtools_advisory.fmtools_python", lambda: "/fake/python")


# ── the verdict object ───────────────────────────────────────────────────────


def test_label_for_returns_the_label_when_present():
    v = AdvisoryVerdict({"a": "review_or_process"}, True)
    assert v.label_for("a") == "review_or_process"


def test_label_for_returns_no_advice_for_an_absent_id():
    """An absent id is NOT a negative -- that conflation is the whole hazard."""
    assert AdvisoryVerdict({}, True).label_for("missing") == "no_advice"


def test_unavailable_label_says_owed_not_agreed():
    text = unavailable_label()
    assert "OWED" in text
    assert "not agreement" in text.lower()


# ── fail-open paths ──────────────────────────────────────────────────────────


def test_empty_input_does_not_run_the_model():
    v = classify_texts({}, labels=LABELS, instruction="x")
    assert v.ran is False and v.labels == {}


def test_missing_venv_fails_open(monkeypatch):
    monkeypatch.setattr("tools.fmtools_advisory.fmtools_python", lambda: None)
    v = classify_texts({"a": "t"}, labels=LABELS, instruction="x")
    assert v.ran is False
    assert "venv" in (v.reason or "")


def test_nonzero_exit_fails_open_and_names_the_code(monkeypatch):
    _stub_run(monkeypatch, result=_Completed(returncode=4, stderr="error: bad label"))
    v = classify_texts({"a": "t"}, labels=LABELS, instruction="x")
    assert v.ran is False
    assert "exit 4" in (v.reason or "")
    assert "bad label" in (v.reason or "")


def test_timeout_fails_open(monkeypatch):
    _stub_run(monkeypatch, raises=subprocess.TimeoutExpired(cmd="x", timeout=1))
    v = classify_texts({"a": "t"}, labels=LABELS, instruction="x", timeout_s=1)
    assert v.ran is False
    assert "timeout" in (v.reason or "")


def test_oserror_fails_open(monkeypatch):
    _stub_run(monkeypatch, raises=OSError("exec format error"))
    v = classify_texts({"a": "t"}, labels=LABELS, instruction="x")
    assert v.ran is False
    assert "subprocess failed" in (v.reason or "")


def test_no_rows_on_stdout_is_not_a_successful_run(monkeypatch):
    """Exit 0 with empty output must not read as 'the model labelled nothing'."""
    _stub_run(monkeypatch, result=_Completed(returncode=0, stdout="\n\n"))
    v = classify_texts({"a": "t"}, labels=LABELS, instruction="x")
    assert v.ran is False
    assert "no rows" in (v.reason or "")


def test_unparseable_lines_are_skipped_without_crashing(monkeypatch):
    stdout = (
        "not json\n"
        + json.dumps({"id": "a", "label": "review_or_process", "ok": True})
        + "\n[1,2]\n"
    )
    _stub_run(monkeypatch, result=_Completed(returncode=0, stdout=stdout))
    v = classify_texts({"a": "t"}, labels=LABELS, instruction="x")
    assert v.ran is True
    assert v.labels == {"a": "review_or_process"}


# ── happy path and per-row errors ────────────────────────────────────────────


def test_labels_are_parsed_and_errors_kept_separate(monkeypatch):
    stdout = (
        json.dumps({"id": "a", "label": "review_or_process", "ok": True})
        + "\n"
        + json.dumps({"id": "b", "label": None, "ok": False, "error": "timeout"})
        + "\n"
    )
    _stub_run(monkeypatch, result=_Completed(returncode=0, stdout=stdout))
    v = classify_texts({"a": "t", "b": "u"}, labels=LABELS, instruction="x")
    assert v.ran is True
    assert v.labels == {"a": "review_or_process"}
    assert v.errors == {"b": "timeout"}
    assert v.label_for("b") == "no_advice", "a failed row must not become a label"


def test_a_row_marked_ok_without_a_string_label_is_an_error_not_a_label(monkeypatch):
    stdout = json.dumps({"id": "a", "label": None, "ok": True}) + "\n"
    _stub_run(monkeypatch, result=_Completed(returncode=0, stdout=stdout))
    v = classify_texts({"a": "t"}, labels=LABELS, instruction="x")
    assert v.labels == {}


def test_the_subprocess_contract_is_the_documented_cli(monkeypatch):
    """Pin the argv shape: this is the fmtools 0.0.219 `classify` contract."""
    capture: dict = {}
    _stub_run(
        monkeypatch,
        result=_Completed(
            returncode=0,
            stdout=json.dumps({"id": "a", "label": "review_or_process", "ok": True}) + "\n",
        ),
        capture=capture,
    )
    classify_texts(
        {"a": "some text"},
        labels=LABELS,
        instruction="decide",
        timeout_s=12.0,
        max_chars=77,
    )
    argv = capture["argv"]
    assert argv[1:4] == ["-m", "fmtools.cli", "classify"]
    assert "--max-chars" in argv and "77" in argv
    for label in LABELS:
        assert label in argv
    assert argv[argv.index("--instruction") + 1] == "decide"
    assert capture["timeout"] == 12.0
    assert json.loads(capture["input"].strip()) == {"id": "a", "text": "some text"}


# ── the disagreement ledger ──────────────────────────────────────────────────


def test_log_disagreements_writes_one_row_each(tmp_path):
    written = log_disagreements(
        "catalog_344",
        [{"memo": "a.md", "deterministic": "x", "advisory": "y"}],
        repo_root=str(tmp_path),
        log_path="state/log.jsonl",
    )
    assert written == 1
    rows = [
        json.loads(line)
        for line in (tmp_path / "state/log.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert rows[0]["lane"] == "catalog_344"
    assert rows[0]["memo"] == "a.md"
    assert "utc" in rows[0]


def test_log_disagreements_appends_rather_than_truncating(tmp_path):
    for memo in ("a.md", "b.md"):
        log_disagreements(
            "l", [{"memo": memo}], repo_root=str(tmp_path), log_path="s/log.jsonl"
        )
    assert len((tmp_path / "s/log.jsonl").read_text().strip().splitlines()) == 2


def test_log_disagreements_is_a_no_op_on_empty_rows(tmp_path):
    assert log_disagreements("l", [], repo_root=str(tmp_path)) == 0


def test_log_disagreements_never_raises_on_a_bad_path():
    """Observability must never break the thing it observes."""
    assert log_disagreements("l", [{"a": 1}], repo_root="/nonexistent/\0bad") == 0


def test_fmtools_python_env_override_is_honoured(monkeypatch, tmp_path):
    fake = tmp_path / "python"
    fake.write_text("#!/bin/sh\n")
    monkeypatch.setenv("PACT_FMTOOLS_PYTHON", str(fake))
    assert fmtools_python() == str(fake)


def test_fmtools_python_returns_none_when_nothing_exists(monkeypatch):
    monkeypatch.setenv("PACT_FMTOOLS_PYTHON", "/nope/python")
    monkeypatch.setenv("DASH_FM_PYTHON", "/also/nope")
    monkeypatch.setattr("os.path.exists", lambda p: False)
    assert fmtools_python() is None


# ── the Catalog #344 advisory column in the hook ─────────────────────────────


@pytest.fixture
def hook_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_fm3_hook", REPO_ROOT / "tools/preflight_hook.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_advisory_column_is_silent_when_opted_out(hook_module, monkeypatch, capsys):
    monkeypatch.setenv("CANONICAL_EQUATION_FM_ADVISORY", "0")
    monkeypatch.setattr(
        "tools.fmtools_advisory.classify_texts",
        lambda *a, **k: pytest.fail("opt-out must not reach the model"),
    )
    hook_module._canonical_equation_fm_advisory(["a.md"], [])
    assert capsys.readouterr().err == ""


def test_advisory_column_skips_a_batch_over_the_cap(hook_module, monkeypatch):
    monkeypatch.delenv("CANONICAL_EQUATION_FM_ADVISORY", raising=False)
    monkeypatch.setattr(
        "tools.fmtools_advisory.classify_texts",
        lambda *a, **k: pytest.fail("over-cap batch must not reach the model"),
    )
    too_many = [f"m{i}.md" for i in range(hook_module._FM_ADVISORY_MAX_MEMOS + 1)]
    hook_module._canonical_equation_fm_advisory(too_many, [])


def test_advisory_column_ignores_memos_the_gate_already_blocks(hook_module, monkeypatch):
    """No point advising on a memo that is already refused -- it is not silent drift."""
    monkeypatch.delenv("CANONICAL_EQUATION_FM_ADVISORY", raising=False)
    monkeypatch.setattr(
        "tools.fmtools_advisory.classify_texts",
        lambda *a, **k: pytest.fail("blocked memos must not reach the model"),
    )
    hook_module._canonical_equation_fm_advisory(["a.md"], ["a.md"])


def test_advisory_column_is_silent_when_the_lane_did_not_run(
    hook_module, monkeypatch, capsys
):
    monkeypatch.delenv("CANONICAL_EQUATION_FM_ADVISORY", raising=False)
    monkeypatch.setattr(
        "tools.fmtools_advisory.classify_texts",
        lambda *a, **k: AdvisoryVerdict({}, False, "no venv"),
    )
    monkeypatch.setattr(Path, "read_text", lambda self, **k: "body")
    hook_module._canonical_equation_fm_advisory(["a.md"], [])
    assert capsys.readouterr().err == ""


def test_advisory_column_is_silent_when_the_lane_agrees(hook_module, monkeypatch, capsys):
    monkeypatch.delenv("CANONICAL_EQUATION_FM_ADVISORY", raising=False)
    monkeypatch.setattr(
        "tools.fmtools_advisory.classify_texts",
        lambda *a, **k: AdvisoryVerdict({"a.md": "review_or_process"}, True),
    )
    monkeypatch.setattr(Path, "read_text", lambda self, **k: "body")
    hook_module._canonical_equation_fm_advisory(["a.md"], [])
    assert capsys.readouterr().err == ""


def test_advisory_column_reports_a_disagreement_without_blocking(
    hook_module, monkeypatch, capsys, tmp_path
):
    monkeypatch.delenv("CANONICAL_EQUATION_FM_ADVISORY", raising=False)
    monkeypatch.setattr(
        "tools.fmtools_advisory.classify_texts",
        lambda *a, **k: AdvisoryVerdict({"a.md": "states_measured_finding"}, True),
    )
    monkeypatch.setattr(Path, "read_text", lambda self, **k: "body")
    logged: list = []
    monkeypatch.setattr(
        "tools.fmtools_advisory.log_disagreements",
        lambda lane, rows, **k: logged.extend(rows) or len(rows),
    )
    result = hook_module._canonical_equation_fm_advisory(["a.md"], [])
    err = capsys.readouterr().err
    assert result is None, "the advisory column must return nothing to gate on"
    assert "ADVISORY" in err and "NOT a verdict" in err
    assert "a.md" in err
    assert logged and logged[0]["memo"] == "a.md"


def test_advisory_column_swallows_an_internal_error(hook_module, monkeypatch, capsys):
    monkeypatch.delenv("CANONICAL_EQUATION_FM_ADVISORY", raising=False)

    def boom(*a, **k):
        raise RuntimeError("bridge exploded")

    monkeypatch.setattr("tools.fmtools_advisory.classify_texts", boom)
    monkeypatch.setattr(Path, "read_text", lambda self, **k: "body")
    hook_module._canonical_equation_fm_advisory(["a.md"], [])
    assert capsys.readouterr().err == ""
