"""The secret-name gate on the PAID dispatch path, and its false-positive allowlist.

WHY THIS FILE EXISTS.  ``validate_runtime_upload_file`` fails a fire closed when a runtime
file looks like a credential.  It is the right shape, but its ``"token"`` marker collides
with this domain's core noun (VIDEO/entropy-coding tokens), and the collision has now
blocked a paid row TWICE:

* 2026-08-04 — ``ddm_r7_token_coder.py``, the receiver's video-token entropy coder, blocked
  the first own-vehicle contest-CPU row.
* 2026-08-20 (``ddm_rr7``) — ``f26_split_token_decoder.py``, the ``rr6`` native token-decode
  port, blocked the wall-clock T4 row that adjudicates the CI decode budget.

Between those two incidents the gate and its allowlist had **no test at all**, which is why
the second one arrived with no regression guard.  A recurrence with no test is the signal
this module answers.

WHAT IT PINS.  Not just "the current entries pass" — that would green on an allowlist that
had silently become a pattern exemption.  It pins the CONTRACT: exact basenames only, the
genuine secret names still refuse, and every entry still names a reviewed file that exists
in the repo (so the allowlist cannot rot into a list of names nobody can check).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tac.deploy.modal.auth_eval import (
    RUNTIME_UPLOAD_BASENAME_ALLOWLIST,
    SENSITIVE_RUNTIME_UPLOAD_SUBSTRINGS,
    validate_runtime_upload_file,
)

REPO = Path(__file__).resolve().parents[3]


def _ok(tmp_path: Path, rel: str) -> None:
    """Validate ``rel`` as if it were a real file in an uploaded runtime tree."""
    target = tmp_path / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("# not a credential\n", encoding="utf-8")
    validate_runtime_upload_file(target, rel)


def test_allowlisted_video_token_modules_are_uploadable(tmp_path: Path) -> None:
    """Both measured false positives pass — the decode modules a fire actually needs."""
    for basename in ("ddm_r7_token_coder.py", "f26_split_token_decoder.py"):
        assert basename in RUNTIME_UPLOAD_BASENAME_ALLOWLIST
        _ok(tmp_path, f"runtime/{basename}")


def test_allowlist_did_not_widen_into_a_pattern(tmp_path: Path) -> None:
    """A DIFFERENT 'token' file is still refused.

    This is the control that distinguishes an exact-basename allowlist from a marker that
    was simply deleted. If someone 'fixes' a future collision by dropping ``token`` from
    SENSITIVE_RUNTIME_UPLOAD_SUBSTRINGS, this test goes red.
    """
    assert "token" in SENSITIVE_RUNTIME_UPLOAD_SUBSTRINGS
    with pytest.raises(ValueError, match="secret-looking"):
        _ok(tmp_path, "runtime/some_other_token_helper.py")


def test_allowlist_is_basename_scoped_not_path_scoped(tmp_path: Path) -> None:
    """An allowlisted basename does not launder a sibling: only that exact file passes."""
    with pytest.raises(ValueError, match="secret-looking"):
        _ok(tmp_path, "runtime/f26_split_token_decoder_backup.py")


@pytest.mark.parametrize(
    "rel",
    [
        "runtime/aws_credentials",
        "runtime/my_api_key.txt",
        "runtime/service_secret.json",
        "runtime/id_rsa",
        "runtime/private_key.pem",
    ],
)
def test_genuine_secret_shapes_still_refuse(tmp_path: Path, rel: str) -> None:
    """The gate still does its actual job."""
    with pytest.raises(ValueError, match="secret-looking"):
        _ok(tmp_path, rel)


# Each allowlist entry is the STAGED basename that reaches the uploader, mapped to the
# reviewed repo source it is copied from. The two differ whenever the stager renames on the
# way in, which is why this cannot be a basename-anywhere search: f26_split_token_decoder.py
# exists only inside a staged candidate tree on external storage, never in the repo under
# that name. (Caught by this very test on first run — the naive search reported the live
# rr7 entry as stale.)
REVIEWED_ALLOWLIST_SOURCES = {
    "ddm_r7_token_coder.py": "ddm_r7_token_coder.py",
    "f26_split_token_decoder.py": "experiments/ddm_wc2c_split_token_decoder.py",
}


def test_every_allowlist_entry_maps_to_a_reviewed_repo_source() -> None:
    """Anti-rot: an allowlist entry nobody can locate is an unreviewable exemption.

    Each entry is admitted on the claim that it is a reviewed in-repo source file with no
    secrets. That claim is checkable only while the source is findable, so this asserts it
    rather than trusting the comment beside it.
    """
    undeclared = sorted(set(RUNTIME_UPLOAD_BASENAME_ALLOWLIST) - set(REVIEWED_ALLOWLIST_SOURCES))
    assert not undeclared, (
        f"allowlist entries {undeclared} carry no declared repo source. Add the staged-name "
        f"-> repo-source mapping here so the 'reviewed, no secrets' claim stays checkable."
    )

    for staged_name, source_hint in REVIEWED_ALLOWLIST_SOURCES.items():
        if staged_name not in RUNTIME_UPLOAD_BASENAME_ALLOWLIST:
            continue  # entry retired from the allowlist; nothing to keep alive
        candidate = REPO / source_hint
        if not candidate.is_file():
            matches = [
                p
                for p in REPO.rglob(source_hint)
                if ".venv" not in p.parts and "node_modules" not in p.parts
            ]
            assert matches, (
                f"allowlisted {staged_name!r} declares repo source {source_hint!r}, which "
                f"does not exist. Either the source moved (re-point it) or the exemption is "
                f"stale (remove it)."
            )
