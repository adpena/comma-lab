"""Tests for ``check_upstream_pin_no_content_drift`` (task #836, 2026-07-31).

BEHAVIOUR, not constants. The gate's whole value is ONE discrimination — CONTENT drift
inside the pinned ``upstream/`` snapshot must refuse, MODE-ONLY drift must not — because
the measured live state was 35 benign exec-bit strips next to 1 real lockfile edit, and a
gate that refused both would be permanently red on a clean checkout (the #821 lesson: a
permanently-red gate trains readers to ignore the suite).

Every fixture is a REAL nested git repo built in tmp_path, so the test exercises the actual
``git -C upstream`` path rather than a mock of it.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from tac.confound_gates import check_upstream_pin_no_content_drift
from tac.preflight import PreflightError


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


@pytest.fixture
def pinned(tmp_path: Path) -> Path:
    """A repo root containing a committed, CLEAN nested ``upstream/`` snapshot."""
    up = tmp_path / "upstream"
    up.mkdir()
    (up / "evaluate.py").write_text("# scorer authority\nSCORE = 1\n", encoding="utf-8")
    (up / "ffmpeg-new").write_bytes(b"\x00\x01\x02binary\xff")
    _git(up, "init", "-q")
    _git(up, "config", "user.email", "t@t")
    _git(up, "config", "user.name", "t")
    _git(up, "add", "-A")
    _git(up, "commit", "-q", "-m", "pin")
    return tmp_path


def test_clean_pin_has_zero_violations(pinned: Path):
    assert check_upstream_pin_no_content_drift(repo_root=pinned, verbose=False) == []


def test_mode_only_change_is_NOT_drift(pinned: Path):
    """THE discrimination. 35 of the 36 measured live entries were exactly this."""
    target = pinned / "upstream" / "evaluate.py"
    before = target.read_bytes()
    os.chmod(target, 0o755)  # flip the exec bit; content untouched
    assert target.read_bytes() == before, "fixture must not alter content"
    # git must actually SEE it as dirty, else the test proves nothing
    out = subprocess.run(
        ["git", "-C", str(pinned / "upstream"), "status", "--porcelain"],
        capture_output=True, text=True, check=True,
    ).stdout
    assert "evaluate.py" in out, "fixture failed to produce a mode-only dirty entry"

    assert check_upstream_pin_no_content_drift(repo_root=pinned, verbose=False) == []


def test_content_change_IS_drift(pinned: Path):
    (pinned / "upstream" / "evaluate.py").write_text("SCORE = 2\n", encoding="utf-8")
    v = check_upstream_pin_no_content_drift(repo_root=pinned, verbose=False)
    assert len(v) == 1 and "CONTENT differs from the pin" in v[0]


def test_binary_content_change_IS_drift(pinned: Path):
    """`git diff --numstat` prints `-` for binaries and cannot decide this; byte-compare can."""
    (pinned / "upstream" / "ffmpeg-new").write_bytes(b"\x00\x01\x02BINARY-EDITED\xff")
    v = check_upstream_pin_no_content_drift(repo_root=pinned, verbose=False)
    assert len(v) == 1 and "ffmpeg-new" in v[0]


def test_deleted_tracked_file_IS_drift(pinned: Path):
    """The 2 lost libSvtAv1Enc symlinks were exactly this shape."""
    (pinned / "upstream" / "evaluate.py").unlink()
    v = check_upstream_pin_no_content_drift(repo_root=pinned, verbose=False)
    assert len(v) == 1 and "DELETED" in v[0]


def test_untracked_file_IS_drift(pinned: Path):
    (pinned / "upstream" / "stray_scratch.json").write_text("{}", encoding="utf-8")
    v = check_upstream_pin_no_content_drift(repo_root=pinned, verbose=False)
    assert len(v) == 1 and "UNTRACKED" in v[0]


def test_absent_nested_repo_fails_open(tmp_path: Path):
    """No upstream/.git -> cannot verify; must not fabricate a pass-with-confidence."""
    assert check_upstream_pin_no_content_drift(repo_root=tmp_path, verbose=False) == []


def test_strict_raises_on_content_drift(pinned: Path):
    (pinned / "upstream" / "evaluate.py").write_text("SCORE = 3\n", encoding="utf-8")
    with pytest.raises(PreflightError):
        check_upstream_pin_no_content_drift(repo_root=pinned, strict=True, verbose=False)


def test_waiver_file_suppresses(pinned: Path):
    (pinned / "upstream" / "evaluate.py").write_text("SCORE = 4\n", encoding="utf-8")
    state = pinned / ".omx" / "state"
    state.mkdir(parents=True)
    (state / "upstream_pin_waiver.txt").write_text(
        "# UPSTREAM_PIN_DRIFT_OK: operator-approved intake of a newer pin 2026-07-31\n",
        encoding="utf-8",
    )
    assert check_upstream_pin_no_content_drift(repo_root=pinned, verbose=False) == []


def test_placeholder_waiver_does_NOT_suppress(pinned: Path):
    """A waiver needs a real rationale (>=4 chars), per the Catalog #287 sister discipline."""
    (pinned / "upstream" / "evaluate.py").write_text("SCORE = 5\n", encoding="utf-8")
    state = pinned / ".omx" / "state"
    state.mkdir(parents=True)
    (state / "upstream_pin_waiver.txt").write_text(
        "# UPSTREAM_PIN_DRIFT_OK: x\n", encoding="utf-8"
    )
    assert check_upstream_pin_no_content_drift(repo_root=pinned, verbose=False) != []


def test_live_repo_is_clean(tmp_path: Path):
    """Strict-flip proof: the REAL upstream/ pin must be content-clean right now."""
    repo = Path(__file__).resolve().parents[3]
    if not (repo / "upstream" / ".git").exists():
        pytest.skip("no nested upstream repo in this checkout")
    v = check_upstream_pin_no_content_drift(repo_root=repo, verbose=False)
    assert v == [], f"live upstream content drift: {v[:3]}"
