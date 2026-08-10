# SPDX-License-Identifier: MIT
"""Tests for tac.deploy.modal.locked_env_probe.

The probe runs INSIDE a Modal container, so these tests exercise it the only way
that is honest offline: substitute the subprocess layer and assert the receipt
the dispatchers will read. The empirical anchor is the 2026-08-10 refusal —
a locked interpreter that exists but cannot exec, and an image interpreter whose
torch differs from the lock.
"""
from __future__ import annotations

import json

import pytest

from tac.deploy.modal import locked_env_probe as lep

# The locked venv lives OUTSIDE the pinned snapshot (see the module docstring: uv's
# .venv/lib64 symlink broke the canonical upstream-snapshot hasher on 2026-08-10).
LOCKED_VENV_PY = "/opt/upstream-locked-venv/bin/python"

LOCKED = {
    "python": "3.12.7",
    "executable": LOCKED_VENV_PY,
    "torch": "2.9.0+cu128",
    "torchvision": "0.24.0",
    "timm": "1.0.22",
    "numpy": "2.3.4",
}
IMAGE = {
    "python": "3.12.7",
    "executable": "/usr/local/bin/python",
    "torch": "2.5.1",
    "torchvision": "0.20.1",
    "timm": "1.0.27",
    "numpy": "1.26.4",
}


def _fake_run(script):
    """Build a _run substitute from a callable mapping (argv) -> (rc, out, err)."""

    def _run(argv, timeout):  # noqa: ARG001 - timeout unused in the fake
        return script(argv)

    return _run


def _default_script(*, locked_rc=0, evaluate_rc=0, locked_env=LOCKED):
    def script(argv):
        exe, _flag, code = argv
        if "evaluate-imports-ok" in code:
            if evaluate_rc != 0:
                return evaluate_rc, "", "ModuleNotFoundError: No module named 'nvidia'"
            return 0, "evaluate-imports-ok\n", ""
        # Match the locked interpreter EXACTLY. Substring matching on ".venv/bin/python"
        # would also catch this test process's own sys.executable, silently aliasing the
        # two interpreters and hiding the very contrast the probe exists to report.
        if exe == LOCKED_VENV_PY:
            if locked_rc != 0:
                return locked_rc, "", "OSError: [Errno 8] Exec format error"
            return 0, json.dumps(locked_env) + "\n", ""
        return 0, json.dumps(IMAGE) + "\n", ""

    return script


def test_parity_packages_match_the_gate():
    """The probe must report exactly what contest_auth_eval.py compares."""

    assert lep.PARITY_PACKAGES == ("torch", "torchvision", "timm", "numpy")


def test_last_json_takes_the_final_object_after_noise():
    stdout = "downloading...\n{\"a\": 1}\nwarning: stale cache\n{\"b\": 2}\n"
    assert lep._last_json(stdout) == {"b": 2}


def test_last_json_skips_unparseable_brace_lines():
    stdout = "{not json at all\n{\"ok\": true}\n{also not json\n"
    assert lep._last_json(stdout) == {"ok": True}


def test_last_json_returns_none_when_absent():
    assert lep._last_json("no objects here\n") is None
    assert lep._last_json("") is None


def test_healthy_env_is_ok_and_never_a_score_claim(monkeypatch):
    monkeypatch.setattr(lep, "_run", _fake_run(_default_script()))

    receipt = lep.probe_locked_upstream_env(
        "/workspace/pact/upstream",
        venv_python=LOCKED_VENV_PY,
        expect_dali=True,
    )

    assert receipt["ok"] is True
    assert receipt["locked_rc"] == 0
    assert receipt["evaluate_import_rc"] == 0
    assert receipt["missing_parity_packages"] == []
    assert receipt["locked_env"]["torch"] == "2.9.0+cu128"
    # The 2026-08-10 contrast the refusal was about, recorded but not adjudicated here.
    assert receipt["image_env"]["torch"] == "2.5.1"
    assert receipt["score_claim"] is False
    assert receipt["scorer_invoked"] is False
    assert receipt["schema"] == "modal_locked_upstream_env_probe.v1"


def test_mach_o_interpreter_is_not_ok_and_keeps_stderr(monkeypatch):
    """The literal 2026-08-10 failure: the file exists, exec raises Errno 8."""

    monkeypatch.setattr(lep, "_run", _fake_run(_default_script(locked_rc=1)))

    receipt = lep.probe_locked_upstream_env(
        "/workspace/pact/upstream",
        venv_python=LOCKED_VENV_PY,
        expect_dali=True,
    )

    assert receipt["ok"] is False
    assert receipt["locked_env"] is None
    assert receipt["missing_parity_packages"] == list(lep.PARITY_PACKAGES)
    assert "Exec format error" in receipt["locked_stderr_tail"]


def test_evaluate_import_failure_blocks_ok(monkeypatch):
    """A venv that execs but cannot import upstream would kill a paid row mid-flight."""

    monkeypatch.setattr(lep, "_run", _fake_run(_default_script(evaluate_rc=1)))

    receipt = lep.probe_locked_upstream_env(
        "/workspace/pact/upstream",
        venv_python=LOCKED_VENV_PY,
        expect_dali=True,
    )

    assert receipt["ok"] is False
    assert receipt["locked_rc"] == 0  # interpreter fine; the IMPORT is what failed
    assert receipt["missing_parity_packages"] == []
    assert "No module named 'nvidia'" in receipt["evaluate_import_stderr_tail"]


def test_partial_parity_report_blocks_ok(monkeypatch):
    partial = dict(LOCKED)
    partial.pop("timm")
    monkeypatch.setattr(lep, "_run", _fake_run(_default_script(locked_env=partial)))

    receipt = lep.probe_locked_upstream_env(
        "/workspace/pact/upstream",
        venv_python=LOCKED_VENV_PY,
        expect_dali=True,
    )

    assert receipt["ok"] is False
    assert receipt["missing_parity_packages"] == ["timm"]


@pytest.mark.parametrize("expect_dali", [True, False])
def test_dali_probe_is_axis_specific(monkeypatch, expect_dali):
    """cu128 carries nvidia-dali-cuda120; the cpu group does not. Probe must differ."""

    seen: list[str] = []

    def script(argv):
        seen.append(argv[2])
        if "evaluate-imports-ok" in argv[2]:
            return 0, "evaluate-imports-ok\n", ""
        return 0, json.dumps(LOCKED) + "\n", ""

    monkeypatch.setattr(lep, "_run", _fake_run(script))

    receipt = lep.probe_locked_upstream_env(
        "/workspace/pact/upstream",
        venv_python=LOCKED_VENV_PY,
        expect_dali=expect_dali,
    )

    evaluate_code = [c for c in seen if "evaluate-imports-ok" in c][0]
    assert ("nvidia.dali" in evaluate_code) is expect_dali
    assert receipt["expect_dali"] is expect_dali


def test_upstream_dir_is_quoted_into_the_probe_source(monkeypatch):
    """A path with a quote or space must not break the generated -c program."""

    seen: list[str] = []

    def script(argv):
        seen.append(argv[2])
        if "evaluate-imports-ok" in argv[2]:
            return 0, "evaluate-imports-ok\n", ""
        return 0, json.dumps(LOCKED) + "\n", ""

    monkeypatch.setattr(lep, "_run", _fake_run(script))
    odd_dir = "/work space/pact's/upstream"

    receipt = lep.probe_locked_upstream_env(
        odd_dir,
        venv_python=LOCKED_VENV_PY,
        expect_dali=False,
    )

    evaluate_code = [c for c in seen if "evaluate-imports-ok" in c][0]
    compile(evaluate_code, "<probe>", "exec")  # the generated program must parse
    assert receipt["upstream_dir"] == odd_dir


def test_venv_python_is_taken_verbatim_never_derived_from_upstream_dir(monkeypatch):
    """The 2026-08-10 regression: deriving the venv from upstream_dir put it INSIDE the
    pinned snapshot, whose hasher refuses the .venv/lib64 symlink. The caller decides."""

    seen: list[str] = []

    def script(argv):
        seen.append(argv[0])
        if "evaluate-imports-ok" in argv[2]:
            return 0, "evaluate-imports-ok\n", ""
        return 0, json.dumps(LOCKED) + "\n", ""

    monkeypatch.setattr(lep, "_run", _fake_run(script))

    receipt = lep.probe_locked_upstream_env(
        "/workspace/pact/upstream",
        venv_python="/opt/elsewhere/bin/python",
        expect_dali=False,
    )

    assert receipt["venv_python"] == "/opt/elsewhere/bin/python"
    assert "/opt/elsewhere/bin/python" in seen
    assert not any(exe.startswith("/workspace/pact/upstream/.venv") for exe in seen)
