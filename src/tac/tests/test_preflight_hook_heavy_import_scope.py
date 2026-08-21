# SPDX-License-Identifier: MIT
"""Catalog #184 scope extension: no module-scope heavy imports on the hook path.

Task #892. `tools/preflight_hook.py` runs `python -m tac.preflight` on EVERY
commit and push. A module-scope `import torch` in `src/tac/preflight.py` was
therefore paid every time -- including in `--no-codebase` mode, which examines
0 of 27 gates and never reaches a `torch.load`.

The cost was BIMODAL, which is why it read as flaky drift rather than a fixed
tax: 0.48s warm, 43.86s real / 0.44s user cold (blocked faulting torch's ~1 GB
of dylibs back into the page cache, which the fleet's memory pressure evicts).
The hook's own 30s timeout sat inside that gap -- green on every warm run,
rc=124 on every cold one.

These tests pin the two halves that matter: the gate FIRES on a re-introduction
(a refusal gate never shown to fire is untrusted, per the positive-control
discipline) and does NOT fire on the cure (a function-local import).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import _check_184_module_scope_heavy_imports as scan


def _mk(tmp_path: Path, body: str) -> Path:
    (tmp_path / "src" / "tac").mkdir(parents=True, exist_ok=True)
    (tmp_path / "src" / "tac" / "preflight.py").write_text(body, encoding="utf-8")
    return tmp_path


@pytest.mark.parametrize(
    "body",
    [
        "import torch\n",
        "from torch import load\n",
        "import torch.nn.functional as F\n",
        "import mlx.core as mx\n",
        "import cv2\n",
    ],
)
def test_module_scope_heavy_import_fires(tmp_path: Path, body: str) -> None:
    """POSITIVE CONTROL -- the gate must actually refuse the thing it names."""
    v = scan(_mk(tmp_path, body))
    assert len(v) == 1, v
    assert "MODULE SCOPE" in v[0]


def test_function_local_import_is_the_cure_and_never_fires(tmp_path: Path) -> None:
    """The fix must not trip the gate, or the gate would forbid its own remedy."""
    body = "def f(p):\n    import torch\n    return torch.load(p)\n"
    assert scan(_mk(tmp_path, body)) == []


@pytest.mark.parametrize(
    "body",
    [
        # The most likely evasion: a "defensive" guard that still pays in full.
        "try:\n    import torch\nexcept ImportError:\n    torch = None\n",
        "try:\n    import nonexistent_xyz\nexcept ImportError:\n    import torch\n",
        "import os\nif os.environ.get('X'):\n    import torch\n",
        "with open(__file__) as _f:\n    import torch\n",
        "class C:\n    import torch\n",
    ],
)
def test_import_time_but_indented_still_fires(tmp_path: Path, body: str) -> None:
    """Scope is COST, not indentation -- all of these run on `import`."""
    v = scan(_mk(tmp_path, body))
    assert len(v) == 1, (body, v)


def test_same_line_waiver_is_respected(tmp_path: Path) -> None:
    body = "import torch  # HOOK_HEAVY_IMPORT_OK:oracle parity needs it at import\n"
    assert scan(_mk(tmp_path, body)) == []


def test_bare_waiver_token_without_rationale_does_not_waive(tmp_path: Path) -> None:
    """Placeholder-rationale rejection (Catalog #287 sister discipline)."""
    body = "import torch  # HOOK_HEAVY_IMPORT_OK:\n"
    assert len(scan(_mk(tmp_path, body))) == 1


@pytest.mark.parametrize("body", ["import json\n", "import numpy as np\n", "import re\n"])
def test_light_imports_never_fire(tmp_path: Path, body: str) -> None:
    """numpy is deliberately NOT heavy: 0.04s and small enough to stay resident."""
    assert scan(_mk(tmp_path, body)) == []


def test_missing_file_is_a_no_op_not_a_crash(tmp_path: Path) -> None:
    assert scan(tmp_path) == []


def test_live_repo_is_clean() -> None:
    """The landed state must satisfy its own gate (live count 0)."""
    repo_root = Path(__file__).resolve().parents[3]
    assert scan(repo_root) == []


# ---------------------------------------------------------------------------
# WIRING tests (round-1 recursive-adversarial review, 2026-08-02).
#
# The 18 tests above all passed while the gate NEVER RAN on a commit: it lived
# in `check_preflight_hook_codebase_default`, inside the `codebase` scope, and
# the hook emits `--no-codebase` (measured: "examined 0 of 27 declared", rc 0).
# A passing unit test proves the UNIT, never the PLACEMENT. These test the
# placement, which is the thing that was actually broken.
# ---------------------------------------------------------------------------

import ast
import importlib.util
import inspect


def _load_hook():
    root = Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location(
        "_ph_under_test", root / "tools" / "preflight_hook.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_main_actually_calls_the_scan() -> None:
    """THE test that was missing. Fails if the call is ever dropped from main()."""
    hook = _load_hook()
    assert "run_hook_path_heavy_import_scan" in inspect.getsource(hook.main)


def test_scan_step_runs_BEFORE_preflight_so_a_FAILING_preflight_cannot_skip_it() -> None:
    """ROUND-4 CORRECTION of my own round-1 test, which asserted the OPPOSITE order.

    Round 1 reasoned only about the VACUOUS-but-rc0 case (where after is fine) and never
    about the FAILURE case. run_preflight() EARLY-RETURNS on failure, so 'after' meant the
    guard was skipped exactly when preflight failed — and a module-scope heavy import
    (43.86 s cold) is itself a cause of preflight failing at rc=124. The guard for the
    timeout was skipped precisely when the timeout fired. BEFORE satisfies both cases."""
    hook = _load_hook()
    src = inspect.getsource(hook.main)
    assert src.index("run_hook_path_heavy_import_scan") < src.index("rc = run_preflight()")


def test_hook_step_returns_nonzero_when_the_scan_fires(monkeypatch) -> None:
    """POSITIVE CONTROL at the HOOK layer, not just the scan layer."""
    hook = _load_hook()
    import tac.preflight as P

    monkeypatch.setattr(
        P, "_check_184_module_scope_heavy_imports",
        lambda root: ["src/tac/preflight.py:1: MODULE SCOPE heavy import 'torch'"],
    )
    assert hook.run_hook_path_heavy_import_scan() == 1


def test_hook_step_fails_open_loudly_if_the_guard_itself_is_broken(monkeypatch, capsys) -> None:
    """Deliberate fail-OPEN: a broken guard must not block every commit."""
    hook = _load_hook()
    import builtins

    real_import = builtins.__import__

    def _boom(name, *a, **k):
        if name == "tac.preflight":
            raise ImportError("simulated broken guard")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _boom)
    assert hook.run_hook_path_heavy_import_scan() == 0
    assert "UNAVAILABLE" in capsys.readouterr().err


def test_hook_step_passes_on_the_live_clean_repo() -> None:
    hook = _load_hook()
    assert hook.run_hook_path_heavy_import_scan() == 0


def test_hook_step_refuses_to_call_a_missing_target_a_pass(tmp_path, monkeypatch, capsys) -> None:
    """ROUND-2: scan([]) on a MISSING target is byte-identical to a clean scan.

    The round-1 suite blessed that ("missing file is a no-op, not a crash"), so a
    wrong REPO_ROOT would have passed SILENTLY — the vacuity genus one layer under
    the round-1 fix for the vacuity genus. Empty scope is VACUOUS, never a PASS.
    """
    hook = _load_hook()
    monkeypatch.setattr(hook, "REPO_ROOT", tmp_path)
    assert hook.run_hook_path_heavy_import_scan() == 0  # fail-open, consistent
    err = capsys.readouterr().err
    assert "VACUOUS" in err and "NOT a pass" in err
    assert "0 of 1 files examined" in err


def test_hook_step_refuses_to_call_an_UNPARSEABLE_target_a_pass(tmp_path, monkeypatch, capsys) -> None:
    """ROUND-3: the third layer. scan() swallows SyntaxError and returns [] — silently
    identical to clean. Round 2 covered MISSING, not CORRUPT. The cure is a DENOMINATOR,
    not a fourth existence check: prove the target parses, so [] means 'examined 1, found 0'."""
    hook = _load_hook()
    d = tmp_path / "src" / "tac"
    d.mkdir(parents=True)
    (d / "preflight.py").write_text("def broken(:\n", encoding="utf-8")
    monkeypatch.setattr(hook, "REPO_ROOT", tmp_path)
    assert hook.run_hook_path_heavy_import_scan() == 0
    err = capsys.readouterr().err
    assert "VACUOUS" in err and "does not parse" in err and "0 of 1 files examined" in err


def test_hook_step_reports_a_real_denominator_on_the_live_repo(capsys) -> None:
    """The live target must actually parse — i.e. a clean run is examined-1-found-0."""
    hook = _load_hook()
    assert hook.run_hook_path_heavy_import_scan() == 0
    assert "VACUOUS" not in capsys.readouterr().err


def test_hook_step_survives_the_scan_itself_raising(monkeypatch, capsys) -> None:
    """ROUND-5: scan() raising propagated UNCAUGHT and crashed the whole hook — a bug
    INSIDE the guard would have blocked every commit in the repo. Symmetry with the other
    two fail-opens: a broken guard fails OPEN and LOUD wherever it breaks."""
    hook = _load_hook()
    import tac.preflight as P

    def _boom(_root):
        raise RuntimeError("simulated scan bug")

    monkeypatch.setattr(P, "_check_184_module_scope_heavy_imports", _boom)
    assert hook.run_hook_path_heavy_import_scan() == 0
    err = capsys.readouterr().err
    assert "CRASHED" in err and "NOT a pass" in err and "simulated scan bug" in err


def test_guard_runs_even_when_preflight_FAILS_end_to_end(monkeypatch) -> None:
    """ROUND-5 EXECUTION control (not source order): drive main() with a FAILING preflight
    and assert the guard already ran. Source-order asserts can be satisfied by dead code."""
    hook = _load_hook()
    order: list[str] = []
    monkeypatch.setattr(hook, "_staged_py_files", lambda: [])
    monkeypatch.setattr(hook, "run_ruff_undefined_name", lambda s: (order.append("ruff"), 0)[1])
    monkeypatch.setattr(hook, "run_hook_path_heavy_import_scan", lambda: (order.append("scan"), 0)[1])
    monkeypatch.setattr(hook, "run_preflight", lambda: (order.append("preflight"), 1)[1])
    monkeypatch.setattr(hook, "run_ci_blind_tests", lambda s: (order.append("ci"), 0)[1])
    monkeypatch.setattr(hook, "run_review_gate", lambda: (order.append("review"), 0)[1])
    assert hook.main() == 1
    assert order == ["ruff", "scan", "preflight"]


def test_END_TO_END_guard_fires_on_a_real_injected_violation(tmp_path, monkeypatch, capsys) -> None:
    """ROUND-6: the whole guard, on a REAL file, both directions — the control that proves
    the chain (hook -> denominator -> scan -> message -> rc) actually works end to end.

    Ran by hand first and it 'failed' (rc 0) — because I injected at line 5, which is INSIDE
    src/tac/preflight.py's module docstring (lines 2-27), so I inserted a STRING, not an
    import. The control was invalid, not the guard. Redone at genuine module scope it fires.
    Persisted here so it is never a one-off hand-run again (the un-persisted-instrument genus).
    """
    real = Path(__file__).resolve().parents[3] / "src" / "tac" / "preflight.py"
    d = tmp_path / "src" / "tac"
    d.mkdir(parents=True)
    lines = real.read_text(encoding="utf-8").splitlines(True)

    # AST-verified module scope: insert after the __future__/first real import, never in the docstring.
    tree = ast.parse("".join(lines))
    first_real = next(n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom)))
    lines.insert(first_real.lineno, "import torch  # injected\n")
    dirty = "".join(lines)
    assert any(
        isinstance(n, ast.Import) and any(a.name == "torch" for a in n.names)
        for n in ast.parse(dirty).body
    ), "positive control is INVALID: the injection is not module-scope"

    (d / "preflight.py").write_text(dirty, encoding="utf-8")
    monkeypatch.setattr(hook_mod := _load_hook(), "REPO_ROOT", tmp_path)
    assert hook_mod.run_hook_path_heavy_import_scan() == 1
    err = capsys.readouterr().err
    assert "MODULE SCOPE" in err and "43.86s cold" in err

    # NEGATIVE side: the unmodified real file must pass, or the gate is unconditional.
    (d / "preflight.py").write_text(real.read_text(encoding="utf-8"), encoding="utf-8")
    assert hook_mod.run_hook_path_heavy_import_scan() == 0


def test_denominator_is_DERIVED_from_the_scan_target_list_not_hardcoded() -> None:
    """ROUND-8: the denominator must come from the SAME tuple the scan iterates.

    It previously pinned "src/tac/preflight.py" as a literal beside the scan, which walks
    _CHECK_184_HOOK_IMPORT_PATH_MODULES. Equal at len 1 — but a second entry would make the
    hook claim 'examined 1' while the scan covered 2, the second never parse-verified: a
    SILENTLY WRONG denominator, the exact defect a denominator exists to prevent."""
    import inspect as _i

    hook = _load_hook()
    src = _i.getsource(hook.run_hook_path_heavy_import_scan)
    assert "_CHECK_184_HOOK_IMPORT_PATH_MODULES" in src, "denominator not derived from the scan list"
    assert "for rel in targets" in src, "denominator does not iterate the scan list"
    assert "examined != len(targets)" in src, "denominator does not compare against the full list"


# --- ddm_si1 (task #929): an absent gate is not a passing gate --------------


def test_review_gate_refuses_when_the_gate_file_is_absent(tmp_path, capsys) -> None:
    """Executed control: a missing review_gate_hook.py must REFUSE, not return 0.

    It previously returned ``0`` -- byte-identical to a clean review -- so
    deleting or renaming the gate silently retired a CLAUDE.md non-negotiable
    with no output at all.
    """
    hook = _load_hook()
    hook.REPO_ROOT = tmp_path  # no tools/review_gate_hook.py here
    rc = hook.run_review_gate()
    assert rc == 1, "an absent review gate reported a pass"
    assert "REFUSE" in capsys.readouterr().err


def test_review_gate_refuses_when_the_interpreter_is_missing(capsys, monkeypatch) -> None:
    """Executed control: the ddm_ob1 shape -- interpreter absent, report success.

    ``subprocess.run`` raising FileNotFoundError used to return ``0``, i.e. the
    gate not running was encoded exactly like the gate passing.
    """
    hook = _load_hook()

    def _boom(*_a, **_k):
        raise FileNotFoundError(".venv/bin/python")

    # `hook.subprocess` IS the global subprocess module, so a bare attribute write here
    # replaced `subprocess.run` for the WHOLE session and every later test that runs a
    # real process died with this FileNotFoundError (measured 2026-08-21: it took out
    # test_process_liveness::test_zombie_is_dead_not_alive and the process-group-kill
    # grandchild control). monkeypatch restores it at teardown; the assertion is unchanged.
    monkeypatch.setattr(hook.subprocess, "run", _boom)
    rc = hook.run_review_gate()
    assert rc == 1, "a review gate that could not launch reported a pass"
    assert "did not run" in capsys.readouterr().err
