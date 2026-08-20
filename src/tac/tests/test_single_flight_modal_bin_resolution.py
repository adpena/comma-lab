"""The single-flight cloud cross-check must actually find the modal CLI.

MEASURED 2026-08-20 (``ddm_rr7``). ``cloud_live_modal_apps`` gated itself on
``shutil.which("modal")``. In this repo ``modal`` is installed at ``<repo>/.venv/bin/modal``
and ``.venv/bin`` is not on PATH, so the lookup returned None and the cloud leg SKIPPED on
every dispatch — announcing it with a single stderr line that scrolls past inside Modal's own
image-build output.

Why that is worth a dedicated test rather than a comment: the skip is FAIL-OPEN by design, so
an inert cloud leg is indistinguishable from a clean one at the call site. It greens by not
looking. When the resolution was fixed, the very first live call reported four running apps
that the local ledgers knew nothing about — i.e. the leg had real signal to give the whole
time.

Scope: resolution only. The liveness predicate (``tasks > 0`` and not stopped) and the refusal
semantics are covered by the guard's own tests; nothing here shells out to Modal.
"""

from __future__ import annotations

import os
from pathlib import Path

from tac.deploy.modal import single_flight

REPO = Path(__file__).resolve().parents[3]


def test_resolves_the_repo_venv_modal_even_when_not_on_path(monkeypatch) -> None:
    """The venv binary wins, and PATH being empty must not hide it."""
    venv_modal = REPO / ".venv" / "bin" / "modal"
    if not venv_modal.is_file():  # pragma: no cover - environment without the venv
        return
    monkeypatch.setattr(os, "environ", {**os.environ, "PATH": ""})
    monkeypatch.setattr(single_flight.shutil, "which", lambda _name: None)

    assert single_flight._resolve_modal_bin() == str(venv_modal)


def test_falls_back_to_path_when_there_is_no_venv_binary(monkeypatch, tmp_path) -> None:
    """A checkout without the venv still uses a PATH-installed modal."""
    monkeypatch.setattr(single_flight, "Path", _PathStub(tmp_path))
    monkeypatch.setattr(single_flight.shutil, "which", lambda _name: "/usr/local/bin/modal")

    assert single_flight._resolve_modal_bin() == "/usr/local/bin/modal"


def test_reports_absent_rather_than_guessing(monkeypatch, tmp_path) -> None:
    """Genuinely absent stays None — the leg is FAIL-OPEN and must not invent a path."""
    monkeypatch.setattr(single_flight, "Path", _PathStub(tmp_path))
    monkeypatch.setattr(single_flight.shutil, "which", lambda _name: None)

    assert single_flight._resolve_modal_bin() is None


class _PathStub:
    """Redirect only ``Path(__file__)`` inside the resolver to a venv-less tree.

    Patching the module's ``Path`` is narrower than relocating the real repo, and keeps the
    two fallback tests from depending on whether the operator's venv happens to exist.
    """

    def __init__(self, root: Path) -> None:
        self._root = root

    def __call__(self, _arg: object) -> Path:
        # parents[4] of <root>/a/b/c/d/e is <root>
        return self._root / "a" / "b" / "c" / "d" / "e"
